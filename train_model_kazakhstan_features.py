import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

RESULTS_PATH = "results.json"
MODEL_NAME = "LSTM+Attention (Kazakhstan, engineered features)"
CLASS_LABELS = ["None", "Mild", "Moderate", "Severe+"]
N_CLASSES = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(SEED)
np.random.seed(SEED)

PROJ_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ = 24, 64, 1, 0.4, 8
BATCH_SIZE, LR, WEIGHT_DECAY, LABEL_SMOOTH = 64, 5e-4, 5e-3, 0.1
MAX_EPOCHS, PATIENCE, MAX_CW = 150, 30, 2.0

def save_results(model_name, metrics, extra=None):
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    entry = {k: float(v) for k, v in metrics.items()}
    entry["timestamp"] = datetime.now().isoformat(timespec="seconds")
    entry["n_classes"] = N_CLASSES
    entry["seed"] = SEED
    if extra:
        entry.update(extra)
    existing = all_results.get(model_name, [])
    if isinstance(existing, dict):
        existing = [existing]
    existing.append(entry)
    all_results[model_name] = existing
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved run #{len(existing)} for '{model_name}' -> {RESULTS_PATH}")

class DroughtDataset(Dataset):
    def __init__(self, X, S, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.S[i], self.y[i]

class LSTMAttention(nn.Module):
    def __init__(self, input_size, soil_size, proj_size, hidden_size,
                 n_layers, dropout, soil_proj, n_classes):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_size), nn.LayerNorm(proj_size),
            nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.lstm = nn.LSTM(proj_size, hidden_size, n_layers, batch_first=True)
        self.attn_w = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.soil_proj = nn.Sequential(
            nn.Linear(soil_size, soil_proj), nn.LayerNorm(soil_proj),
            nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.fc1 = nn.Linear(hidden_size + soil_proj, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x, s):
        B, T, F_ = x.shape
        x = self.proj(x.reshape(B*T, F_)).reshape(B, T, -1)
        out, _ = self.lstm(x)
        scores = self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        ctx = (out * weights.unsqueeze(-1)).sum(dim=1)
        soil_ctx = self.soil_proj(s)
        combined = torch.cat([self.drop(ctx), soil_ctx], dim=1)
        return self.fc2(self.relu(self.fc1(combined)))

def evaluate(model, loader, criterion):
    model.eval()
    loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for xb, sb, yb in loader:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb, sb)
            loss += criterion(logits, yb).item() * len(xb)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    preds, trues = np.array(preds), np.array(trues)
    return (loss / len(loader.dataset),
            accuracy_score(trues, preds),
            f1_score(trues, preds, average="macro", zero_division=0),
            mean_absolute_error(trues, preds),
            np.sqrt(mean_squared_error(trues, preds)),
            preds, trues)

def main():
    if not os.path.exists("sequences_kazakhstan_features.npz"):
        raise SystemExit("sequences_kazakhstan_features.npz not found -- "
                          "run load_data_kazakhstan_features.py first.")
    data = np.load("sequences_kazakhstan_features.npz", allow_pickle=True)
    X_train, y_train, S_train = data["X_train"], data["y_train"], data["S_train"]
    X_val, y_val, S_val = data["X_val"], data["y_val"], data["S_val"]
    X_test, y_test, S_test = data["X_test"], data["y_test"], data["S_test"]
    feature_names = data["feature_names"]

    print(f"Device: {DEVICE} | SEED: {SEED}")
    print(f"Features ({len(feature_names)}): {list(feature_names)}")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Train class dist: {np.bincount(y_train, minlength=N_CLASSES)} ({CLASS_LABELS})")

    train_ld = DataLoader(DroughtDataset(X_train, S_train, y_train), BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(DroughtDataset(X_val, S_val, y_val), BATCH_SIZE*2)
    test_ld = DataLoader(DroughtDataset(X_test, S_test, y_test), BATCH_SIZE*2)

    cw_vals = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = np.clip(np.sqrt(cw_vals), None, MAX_CW).astype(np.float32)
    print(f"Class weights: {np.round(cw, 3)}")
    cw_t = torch.tensor(cw).to(DEVICE)

    model = LSTMAttention(X_train.shape[-1], S_train.shape[-1], PROJ_SIZE,
                           HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ,
                           N_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (MAX_EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, best_state, no_imp = 0.0, None, 0
    print(f"\n{'Epoch':>6} {'Train':>9} {'Val':>9} {'ValAcc':>7} {'ValF1':>7}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tloss = 0.0
        for xb, sb, yb in train_ld:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb, sb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item() * len(xb)
        tloss /= len(train_ld.dataset)
        scheduler.step()

        vloss, vacc, vf1, *_ = evaluate(model, val_ld, criterion)
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {tloss:>9.4f} {vloss:>9.4f} {vacc:>7.4f} {vf1:>7.4f}")

        if vf1 > best_f1:
            best_f1, best_state, no_imp = vf1, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val F1: {best_f1:.4f})")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    torch.save(model.state_dict(), "model_kazakhstan_features.pt")
    print("Saved model_kazakhstan_features.pt")

    _, acc, f1, mae, rmse, preds, trues = evaluate(model, test_ld, criterion)
    print(f"\n{'='*50}\nTEST RESULTS (Kazakhstan, engineered features)\n{'='*50}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    cm = confusion_matrix(trues, preds, labels=list(range(N_CLASSES)))
    print(f"\nConfusion Matrix:\n{cm}")

    save_results(MODEL_NAME, {"Accuracy": acc, "Macro F1": f1, "MAE": mae, "RMSE": rmse},
                 extra={"confusion_matrix": cm.tolist(), "n_features": int(X_train.shape[-1]),
                        "feature_names": list(feature_names), "real_data": True})

    print(f"\nCOMPARISON: original 1-feature Kazakhstan model scored Macro F1 = 0.542")
    print(f"This 14-feature engineered model scored Macro F1 = {f1:.4f}")
    print(f"Difference: {f1 - 0.542:+.4f}")

if __name__ == "__main__":
    main()
