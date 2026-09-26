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
KZ_DATA = "sequences_kazakhstan_features.npz"
PRETRAINED_PATH = "model_us_1feature.pt"
N_CLASSES = 4
CLASS_LABELS = ["None", "Mild", "Moderate", "Severe+"]

PROJ_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT = 24, 64, 1, 0.4
BATCH_SIZE, LR, WEIGHT_DECAY, LABEL_SMOOTH = 64, 5e-4, 5e-3, 0.1
MAX_EPOCHS, PATIENCE, MAX_CW = 80, 20, 2.0
FT_LR_FACTOR = 0.2
FT_MAX_EPOCHS = 60
FT_PATIENCE = 15

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


class LSTMAttentionKZ(nn.Module):
    """KZ features model: weather branch only (input_size=14, PROJ=24, HIDDEN=64)."""
    def __init__(self, input_size, proj_size, hidden_size, n_layers, dropout, n_classes):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_size), nn.LayerNorm(proj_size),
            nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.lstm = nn.LSTM(proj_size, hidden_size, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.attn_w = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, T, F_ = x.shape
        x = self.proj(x.reshape(B * T, F_)).reshape(B, T, -1)
        out, _ = self.lstm(x)
        scores = self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        ctx = (out * weights.unsqueeze(-1)).sum(dim=1)
        return self.fc2(self.relu(self.fc1(self.drop(ctx))))


def evaluate(model, loader, criterion):
    model.eval()
    loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for xb, _, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
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


def train_model(model, train_ld, val_ld, criterion, max_epochs, patience, lr, tag):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    def lr_lambda(epoch):
        warmup = 3
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (max_epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, best_state, no_imp = 0.0, None, 0
    print(f"\n  {'Epoch':>6} {'Train':>9} {'Val':>9} {'ValAcc':>7} {'ValF1':>7}")
    for epoch in range(1, max_epochs + 1):
        model.train()
        tloss = 0.0
        for xb, _, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item() * len(xb)
        tloss /= len(train_ld.dataset)
        scheduler.step()

        vloss, vacc, vf1, *_ = evaluate(model, val_ld, criterion)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  {tag} {epoch:>4} {tloss:>9.4f} {vloss:>9.4f} {vacc:>7.4f} {vf1:>7.4f}")
        if vf1 > best_f1:
            best_f1, best_state, no_imp = vf1, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  {tag} early stopping at epoch {epoch} (best val F1: {best_f1:.4f})")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return best_f1


def print_results(name, model, test_ld, criterion):
    _, acc, f1, mae, rmse, preds, trues = evaluate(model, test_ld, criterion)
    cm = confusion_matrix(trues, preds, labels=list(range(N_CLASSES)))
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    print(f"\n  TEST RESULTS: {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Per-class recall:")
    for lbl, r in zip(CLASS_LABELS, per_class):
        print(f"    {lbl:<10} {r:.4f}")
    return {"Accuracy": acc, "Macro F1": f1, "MAE": mae, "RMSE": rmse}


def main():
    SEED = int(os.environ.get("SEED", 42))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(KZ_DATA):
        raise SystemExit(f"{KZ_DATA} not found -- run load_data_kazakhstan_features.py first.")
    if not os.path.exists(PRETRAINED_PATH):
        raise SystemExit(f"{PRETRAINED_PATH} not found -- run train_model_us_1feature.py first.")

    data = np.load(KZ_DATA, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = data["feature_names"]

    print(f"Device: {DEVICE} | SEED: {SEED}")
    print(f"KZ features: {list(feature_names)} ({X_train.shape[-1]} dims)")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Train class dist: {np.bincount(y_train, minlength=N_CLASSES)}")

    S_dummy = np.zeros((len(X_train), 31), dtype=np.float32)
    S_val_dummy = np.zeros((len(X_val), 31), dtype=np.float32)
    S_test_dummy = np.zeros((len(X_test), 31), dtype=np.float32)

    train_ld = DataLoader(DroughtDataset(X_train, S_dummy, y_train),
                          BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(DroughtDataset(X_val, S_val_dummy, y_val), BATCH_SIZE * 2)
    test_ld = DataLoader(DroughtDataset(X_test, S_test_dummy, y_test), BATCH_SIZE * 2)

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = np.clip(np.sqrt(cw), None, MAX_CW).astype(np.float32)
    cw_t = torch.tensor(cw).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=LABEL_SMOOTH)
    print(f"Class weights: {np.round(cw, 3)}")

    # --- Part 1: From-scratch baseline ---
    print(f"\n{'='*60}")
    print(f"PART 1: KZ features model FROM SCRATCH (random init)")
    print(f"{'='*60}")
    model_scratch = LSTMAttentionKZ(X_train.shape[-1], PROJ_SIZE, HIDDEN_SIZE,
                                    N_LAYERS, DROPOUT, N_CLASSES).to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model_scratch.parameters()):,}")

    train_model(model_scratch, train_ld, val_ld, criterion,
                MAX_EPOCHS, PATIENCE, LR, "SCRATCH")
    torch.save(model_scratch.state_dict(), f"model_kazakhstan_scratch_seed{SEED}.pt")
    metrics_scratch = print_results("KZ features from scratch", model_scratch, test_ld, criterion)
    save_results(f"LSTM+Attention (KZ features, from scratch, seed{SEED})", metrics_scratch,
                 extra={"n_features": int(X_train.shape[-1]), "seed": SEED,
                        "transfer_learning": False})

    # --- Part 2: Transfer learning from US PRECTOT ---
    print(f"\n{'='*60}")
    print(f"PART 2: TRANSFER LEARNING from US PRECTOT model")
    print(f"{'='*60}")

    us_state = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=True)
    print(f"  Loaded US pretrained keys: {list(us_state.keys())}")

    model_ft = LSTMAttentionKZ(X_train.shape[-1], PROJ_SIZE, HIDDEN_SIZE,
                               N_LAYERS, DROPOUT, N_CLASSES).to(DEVICE)

    # Note: proj.0.weight does NOT transfer (US: 24x1 vs KZ: 24x14).
    # All other layers share the same shape (PROJ=24, HIDDEN=64).
    compatible_keys = ["proj.0.bias", "proj.1.weight", "proj.1.bias",
                       "lstm.weight_ih_l0", "lstm.weight_hh_l0",
                       "lstm.bias_ih_l0", "lstm.bias_hh_l0",
                       "attn_w.weight", "attn_w.bias",
                       "attn_v.weight",
                       "fc1.weight", "fc1.bias",
                       "fc2.weight", "fc2.bias"]
    transferred = []
    skipped = []
    for k in compatible_keys:
        if k in us_state and k in model_ft.state_dict():
            us_shape = us_state[k].shape
            kz_shape = model_ft.state_dict()[k].shape
            if us_shape == kz_shape:
                model_ft.state_dict()[k].copy_(us_state[k])
                transferred.append(k)
            else:
                skipped.append(f"{k} ({us_shape} vs {kz_shape})")
        else:
            skipped.append(f"{k} (missing)")

    print(f"  Transferred {len(transferred)} layers: {transferred}")
    if skipped:
        print(f"  Skipped {len(skipped)} layers: {skipped}")

    ft_lr = LR * FT_LR_FACTOR
    print(f"  Fine-tuning LR: {ft_lr:.2e} (factor {FT_LR_FACTOR}x from base LR)")

    train_model(model_ft, train_ld, val_ld, criterion,
                FT_MAX_EPOCHS, FT_PATIENCE, ft_lr, "TRANSFER")
    torch.save(model_ft.state_dict(), f"model_kazakhstan_transfer_seed{SEED}.pt")
    metrics_ft = print_results("KZ features transfer-learned", model_ft, test_ld, criterion)
    save_results(f"LSTM+Attention (KZ features, transfer-learned, seed{SEED})", metrics_ft,
                 extra={"n_features": int(X_train.shape[-1]), "seed": SEED,
                        "transfer_learning": True,
                        "pretrained_source": "US PRECTOT 1-feat",
                        "transferred_layers": transferred,
                        "ft_lr": ft_lr})

    # --- Part 3: Comparison ---
    print(f"\n{'='*60}")
    print(f"TRANSFER LEARNING COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'Macro F1':>9} {'Accuracy':>9}")
    print(f"{'-'*45} {'-'*9} {'-'*9}")
    print(f"{'KZ features, from scratch':<45} {metrics_scratch['Macro F1']:>9.4f} {metrics_scratch['Accuracy']:>9.4f}")
    print(f"{'KZ features, transfer-learned':<45} {metrics_ft['Macro F1']:>9.4f} {metrics_ft['Accuracy']:>9.4f}")
    print(f"{'KZ features, prior best (scratch, seed=42)':<45} {'0.6285':>9} {'0.8614':>9}")
    delta = metrics_ft["Macro F1"] - metrics_scratch["Macro F1"]
    print(f"\nTransfer learning {'helped' if delta > 0 else 'did not help'}: "
          f"{'+' if delta > 0 else ''}{delta:.4f} Macro F1")


if __name__ == "__main__":
    main()
