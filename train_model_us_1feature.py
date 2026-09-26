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
DATA_PATH = "sequences_4.npz"
PRECTOT_IDX = 0
N_CLASSES = 4
CLASS_LABELS = ["None", "Mild", "Moderate", "Severe+"]

PROJ_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ = 24, 64, 1, 0.4, 8
BATCH_SIZE, LR, WEIGHT_DECAY, LABEL_SMOOTH = 256, 5e-4, 5e-3, 0.1
MAX_EPOCHS, PATIENCE, MAX_CW = 50, 20, 2.0
MAX_TRAIN_SAMPLES = 200000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class LSTMAttentionWithSoil(nn.Module):
    """Two-branch model (weather + soil) -- used for Variants A and B."""
    def __init__(self, input_size, soil_size, proj_size, hidden_size,
                 n_layers, dropout, soil_proj, n_classes):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_size), nn.LayerNorm(proj_size),
            nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.lstm = nn.LSTM(proj_size, hidden_size, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
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


class LSTMAttentionWeatherOnly(nn.Module):
    """Single-branch model (weather only) -- used for Variant C."""
    def __init__(self, input_size, proj_size, hidden_size,
                 n_layers, dropout, n_classes):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_size), nn.LayerNorm(proj_size),
            nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.lstm = nn.LSTM(proj_size, hidden_size, n_layers,
                            batch_first=True,
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


def evaluate_with_soil(model, loader, criterion):
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


def evaluate_weather_only(model, loader, criterion):
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


def train_with_soil(model, train_ld, val_ld, test_ld, criterion, model_name, model_path):
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

        vloss, vacc, vf1, *_ = evaluate_with_soil(model, val_ld, criterion)
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {tloss:>9.4f} {vloss:>9.4f} {vacc:>7.4f} {vf1:>7.4f}")
        if vf1 > best_f1:
            best_f1, best_state, no_imp = vf1, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val F1: {best_f1:.4f})")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    torch.save(model.state_dict(), model_path)
    print(f"  Saved {model_path}")

    _, acc, f1, mae, rmse, preds, trues = evaluate_with_soil(model, test_ld, criterion)
    cm = confusion_matrix(trues, preds, labels=list(range(N_CLASSES)))
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    print(f"\n  TEST RESULTS: {model_name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Per-class recall:")
    for lbl, r in zip(CLASS_LABELS, per_class):
        print(f"    {lbl:<10} {r:.4f}")

    return {"Accuracy": acc, "Macro F1": f1, "MAE": mae, "RMSE": rmse}


def train_weather_only(model, train_ld, val_ld, test_ld, criterion, model_name, model_path):
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

        vloss, vacc, vf1, *_ = evaluate_weather_only(model, val_ld, criterion)
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {tloss:>9.4f} {vloss:>9.4f} {vacc:>7.4f} {vf1:>7.4f}")
        if vf1 > best_f1:
            best_f1, best_state, no_imp = vf1, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val F1: {best_f1:.4f})")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    torch.save(model.state_dict(), model_path)
    print(f"  Saved {model_path}")

    _, acc, f1, mae, rmse, preds, trues = evaluate_weather_only(model, test_ld, criterion)
    cm = confusion_matrix(trues, preds, labels=list(range(N_CLASSES)))
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    print(f"\n  TEST RESULTS: {model_name}")
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
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} not found -- run load_data.py first.")

    data = np.load(DATA_PATH, allow_pickle=True)
    X_train_all = data["X_train"]   # (N, 90, 18)
    X_val_all = data["X_val"]
    X_test_all = data["X_test"]
    S_train = data["S_train"]
    S_val = data["S_val"]
    S_test = data["S_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    print(f"Device: {DEVICE}")
    print(f"Full data shape: X_train={X_train_all.shape}, S_train={S_train.shape}")
    print(f"Feature 0 (PRECTOT) range: [{X_train_all[:,:,PRECTOT_IDX].min():.3f}, "
          f"{X_train_all[:,:,PRECTOT_IDX].max():.3f}]")

    if len(X_train_all) > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train_all), MAX_TRAIN_SAMPLES, replace=False)
        idx.sort()
        X_train_all = X_train_all[idx]
        S_train = S_train[idx]
        y_train = y_train[idx]
        print(f"Subsampled training to {MAX_TRAIN_SAMPLES} rows for speed")

    print(f"Train: {X_train_all.shape} | Val: {X_val_all.shape} | Test: {X_test_all.shape}")
    print(f"Train class dist: {np.bincount(y_train, minlength=N_CLASSES)}")
    print(f"Architecture: PROJ={PROJ_SIZE}, HIDDEN={HIDDEN_SIZE}, SOIL_PROJ={SOIL_PROJ}")

    results = {}

    # --- Variant A: Full US (18 weather + soil), architecture-matched ---
    print(f"\n{'='*60}")
    print(f"VARIANT A: Full US (18 weather + soil)")
    print(f"{'='*60}")
    X_train_a, X_val_a, X_test_a = X_train_all, X_val_all, X_test_all
    train_ld = DataLoader(DroughtDataset(X_train_a, S_train, y_train),
                          BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(DroughtDataset(X_val_a, S_val, y_val), BATCH_SIZE * 2)
    test_ld = DataLoader(DroughtDataset(X_test_a, S_test, y_test), BATCH_SIZE * 2)

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = np.clip(np.sqrt(cw), None, MAX_CW).astype(np.float32)
    cw_t = torch.tensor(cw).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=LABEL_SMOOTH)

    model_a = LSTMAttentionWithSoil(18, S_train.shape[-1], PROJ_SIZE, HIDDEN_SIZE,
                                    N_LAYERS, DROPOUT, SOIL_PROJ, N_CLASSES).to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model_a.parameters()):,}")

    metrics_a = train_with_soil(model_a, train_ld, val_ld, test_ld, criterion,
                                "LSTM+Attention (US, full 18-feat, arch-matched)",
                                "model_us_full_archmatched.pt")
    save_results("LSTM+Attention (US, full 18-feat, arch-matched)", metrics_a,
                 extra={"n_features": 18, "variant": "A_full"})
    results["A"] = {"name": "Full US (18 feat + soil)", **metrics_a}

    # --- Variant B: US 1-feature PRECTOT + soil ---
    print(f"\n{'='*60}")
    print(f"VARIANT B: US 1-feature PRECTOT + soil")
    print(f"{'='*60}")
    X_train_b = X_train_all[:, :, [PRECTOT_IDX]]
    X_val_b = X_val_all[:, :, [PRECTOT_IDX]]
    X_test_b = X_test_all[:, :, [PRECTOT_IDX]]
    print(f"  Reduced shape: {X_train_b.shape} (1 feature: PRECTOT)")

    train_ld = DataLoader(DroughtDataset(X_train_b, S_train, y_train),
                          BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(DroughtDataset(X_val_b, S_val, y_val), BATCH_SIZE * 2)
    test_ld = DataLoader(DroughtDataset(X_test_b, S_test, y_test), BATCH_SIZE * 2)

    model_b = LSTMAttentionWithSoil(1, S_train.shape[-1], PROJ_SIZE, HIDDEN_SIZE,
                                    N_LAYERS, DROPOUT, SOIL_PROJ, N_CLASSES).to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model_b.parameters()):,}")

    metrics_b = train_with_soil(model_b, train_ld, val_ld, test_ld, criterion,
                                "LSTM+Attention (US, 1-feat PRECTOT + soil)",
                                "model_us_1feat_soil.pt")
    save_results("LSTM+Attention (US, 1-feat PRECTOT + soil)", metrics_b,
                 extra={"n_features": 1, "variant": "B_prectot_soil"})
    results["B"] = {"name": "US PRECTOT + soil", **metrics_b}

    # --- Variant C: US 1-feature PRECTOT only (no soil) ---
    print(f"\n{'='*60}")
    print(f"VARIANT C: US 1-feature PRECTOT only (no soil)")
    print(f"{'='*60}")

    train_ld = DataLoader(DroughtDataset(X_train_b, S_train, y_train),
                          BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(DroughtDataset(X_val_b, S_val, y_val), BATCH_SIZE * 2)
    test_ld = DataLoader(DroughtDataset(X_test_b, S_test, y_test), BATCH_SIZE * 2)

    model_c = LSTMAttentionWeatherOnly(1, PROJ_SIZE, HIDDEN_SIZE,
                                       N_LAYERS, DROPOUT, N_CLASSES).to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model_c.parameters()):,}")

    metrics_c = train_weather_only(model_c, train_ld, val_ld, test_ld, criterion,
                                   "LSTM+Attention (US, 1-feat PRECTOT)",
                                   "model_us_1feature.pt")
    save_results("LSTM+Attention (US, 1-feat PRECTOT)", metrics_c,
                 extra={"n_features": 1, "variant": "C_prectot_only"})
    results["C"] = {"name": "US PRECTOT only", **metrics_c}

    # --- Comparison Table ---
    print(f"\n{'='*72}")
    print(f"TASK 1(c) COMPARISON: CONTROL EXPERIMENT RESULTS")
    print(f"{'='*72}")
    print(f"{'Model':<45} {'Macro F1':>9} {'Accuracy':>9}")
    print(f"{'-'*45} {'-'*9} {'-'*9}")
    for key in ["A", "B", "C"]:
        r = results[key]
        print(f"US {r['name']:<41} {r['Macro F1']:>9.4f} {r['Accuracy']:>9.4f}")
    print(f"{'-'*45} {'-'*9} {'-'*9}")
    print(f"{'KZ, 1 raw SPEI (from prior run)':<45} {'0.5420':>9} {'0.8280':>9}")
    print(f"{'KZ, 14 engineered features (from prior run)':<45} {'0.6285':>9} {'0.8614':>9}")
    print(f"\nKey question: 'if the US had as few features as Kazakhstan,")
    print(f"would the quality gap be the same?'")
    print(f"  US 1-feature PRECTOT F1 = {results['C']['Macro F1']:.4f}")
    print(f"  KZ 1-feature SPEI F1    = 0.5420")
    print(f"  Gap: {results['C']['Macro F1'] - 0.542:+.4f}")


if __name__ == "__main__":
    main()
