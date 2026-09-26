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
DATA_PATH = "sequences_kazakhstan_features.npz"
N_CLASSES = 4
CLASS_LABELS = ["None", "Mild", "Moderate", "Severe+"]

PROJ_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ = 24, 64, 1, 0.4, 8
BATCH_SIZE, LR, WEIGHT_DECAY, LABEL_SMOOTH = 64, 5e-4, 5e-3, 0.1
MAX_EPOCHS, PATIENCE, MAX_CW = 150, 30, 2.0

SEEDS = [42, 123, 2026]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_results(model_name, metrics, extra=None):
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    entry = {k: float(v) if not isinstance(v, list) else v for k, v in metrics.items()}
    entry["timestamp"] = datetime.now().isoformat(timespec="seconds")
    entry["n_classes"] = N_CLASSES
    if extra:
        entry.update(extra)
    existing = all_results.get(model_name, [])
    if isinstance(existing, dict):
        existing = [existing]
    existing.append(entry)
    all_results[model_name] = existing
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved run #{len(existing)} for '{model_name}' -> {RESULTS_PATH}")


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
        x = self.proj(x.reshape(B * T, F_)).reshape(B, T, -1)
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


def train_and_evaluate(seed, X_train, y_train, S_train, X_val, y_val, S_val,
                        X_test, y_test, S_test, cw_t, criterion):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ld = DataLoader(DroughtDataset(X_train, S_train, y_train),
                          BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(DroughtDataset(X_val, S_val, y_val), BATCH_SIZE * 2)
    test_ld = DataLoader(DroughtDataset(X_test, S_test, y_test), BATCH_SIZE * 2)

    model = LSTMAttention(X_train.shape[-1], S_train.shape[-1], PROJ_SIZE,
                          HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ,
                          N_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (MAX_EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, best_state, no_imp = 0.0, None, 0
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
        if epoch % 30 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}: train={tloss:.4f} val={vloss:.4f} "
                  f"valAcc={vacc:.4f} valF1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1, best_state, no_imp = vf1, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    Early stopping at epoch {epoch} (best val F1: {best_f1:.4f})")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model_path = f"model_kazakhstan_features_seed{seed}.pt"
    torch.save(model.state_dict(), model_path)

    _, acc, f1, mae, rmse, preds, trues = evaluate(model, test_ld, criterion)
    cm = confusion_matrix(trues, preds, labels=list(range(N_CLASSES)))
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    print(f"    Test Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(f"    Per-class recall:")
    for lbl, r in zip(CLASS_LABELS, per_class):
        print(f"      {lbl:<10} {r:.4f}")

    return {"Accuracy": acc, "Macro F1": f1, "MAE": mae, "RMSE": rmse,
            "per_class_recall": per_class.tolist()}


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} not found -- run load_data_kazakhstan_features.py first.")

    data = np.load(DATA_PATH, allow_pickle=True)
    X_train, y_train, S_train = data["X_train"], data["y_train"], data["S_train"]
    X_val, y_val, S_val = data["X_val"], data["y_val"], data["S_val"]
    X_test, y_test, S_test = data["X_test"], data["y_test"], data["S_test"]
    feature_names = data["feature_names"]

    print(f"Device: {DEVICE}")
    print(f"Data: {list(feature_names)} ({X_train.shape[-1]} features)")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Train class dist: {np.bincount(y_train, minlength=N_CLASSES)}")

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = np.clip(np.sqrt(cw), None, MAX_CW).astype(np.float32)
    cw_t = torch.tensor(cw).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=LABEL_SMOOTH)
    print(f"Class weights: {np.round(cw, 3)}")

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        metrics = train_and_evaluate(
            seed, X_train, y_train, S_train, X_val, y_val, S_val,
            X_test, y_test, S_test, cw_t, criterion)

        save_results("LSTM+Attention (Kazakhstan, engineered features, multi-seed)",
                     metrics,
                     extra={"n_features": int(X_train.shape[-1]),
                            "seed": seed,
                            "feature_names": list(feature_names)})
        all_results.append({"seed": seed, **metrics})

    # --- Summary Table ---
    print(f"\n{'='*72}")
    print(f"TASK 2: MULTI-SEED ROBUSTNESS RESULTS")
    print(f"{'='*72}")
    print(f"{'Seed':>6} {'Accuracy':>9} {'Macro F1':>9} {'MAE':>9} {'RMSE':>9}")
    print(f"{'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for r in all_results:
        print(f"{r['seed']:>6} {r['Accuracy']:>9.4f} {r['Macro F1']:>9.4f} "
              f"{r['MAE']:>9.4f} {r['RMSE']:>9.4f}")

    accs = [r["Accuracy"] for r in all_results]
    f1s = [r["Macro F1"] for r in all_results]
    print(f"{'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    print(f"{'Mean':>6} {np.mean(accs):>9.4f} {np.mean(f1s):>9.4f}")
    print(f"{'Std':>6} {np.std(accs):>9.4f} {np.std(f1s):>9.4f}")
    print(f"{'Range':>6} {max(accs)-min(accs):>9.4f} {max(f1s)-min(f1s):>9.4f}")

    print(f"\nCONCLUSION:")
    print(f"  Macro F1 range: {min(f1s):.4f} - {max(f1s):.4f} (spread: {max(f1s)-min(f1s):.4f})")
    print(f"  Accuracy range: {min(accs):.4f} - {max(accs):.4f} (spread: {max(accs)-min(accs):.4f})")
    if max(f1s) - min(f1s) < 0.03:
        print(f"  -> Results are STABLE across seeds (spread < 0.03)")
    else:
        print(f"  -> Results show MODERATE variance across seeds (spread >= 0.03)")


if __name__ == "__main__":
    main()
