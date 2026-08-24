import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

RESULTS_PATH = "results.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_SCHEME = 4          # <-- must match the sequences_{N}.npz used by train_no_none.py
DATA_PATH = f"sequences_{CLASS_SCHEME}.npz"
N_CLASSES_OUT = CLASS_SCHEME - 1   # one fewer class, None dropped
BATCH_SIZE = 256
SEED = 42

FULL_LABELS = {
    4: ["None", "Mild", "Moderate", "Severe+"],
    5: ["None", "D0", "D1", "D2", "D3+"],
    6: ["None", "D0", "D1", "D2", "D3", "D4"],
}[CLASS_SCHEME]
OUT_LABELS = FULL_LABELS[1:]

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
    entry["n_classes"] = N_CLASSES_OUT
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

def drop_none_and_remap(X, S, y):
    mask = y != 0
    return X[mask], S[mask], y[mask] - 1

# ------------------------------------------------------------------ #
# Dataset (weather sequence + static soil vector)
# ------------------------------------------------------------------ #
class DroughtDataset(Dataset):
    def __init__(self, X, S, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.S[i], self.y[i]

# ------------------------------------------------------------------ #
# LSTM-only model (no attention), same soil branch for parity
# ------------------------------------------------------------------ #
class LSTMOnly(nn.Module):
    def __init__(self, input_size, soil_size, hidden_size, soil_proj, n_classes, dropout=0.4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(dropout*0.5))
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.soil_proj = nn.Sequential(
            nn.Linear(soil_size, soil_proj), nn.LayerNorm(soil_proj), nn.ReLU(), nn.Dropout(dropout*0.5))
        self.fc1 = nn.Linear(hidden_size + soil_proj, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x, s):
        B, T, F = x.shape
        x = self.proj(x.reshape(B*T, F)).reshape(B, T, -1)
        _, (h, _) = self.lstm(x)
        soil_ctx = self.soil_proj(s)
        combined = torch.cat([self.drop(h[-1]), soil_ctx], dim=1)
        return self.fc2(self.relu(self.fc1(combined)))

def train_lstm_only(X_train, S_train, y_train, X_val, S_val, y_val, X_test, S_test, y_test):
    print("\n" + "="*50)
    print("BASELINE 2: LSTM-only (no attention), 'None' excluded")
    print("="*50)

    train_ld = DataLoader(DroughtDataset(X_train, S_train, y_train), BATCH_SIZE, shuffle=True)
    val_ld   = DataLoader(DroughtDataset(X_val, S_val, y_val), BATCH_SIZE*2)
    test_ld  = DataLoader(DroughtDataset(X_test, S_test, y_test), BATCH_SIZE*2)

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = np.clip(np.sqrt(cw), None, 2.0).astype(np.float32)
    cw_t = torch.tensor(cw).to(DEVICE)

    model = LSTMOnly(X_train.shape[-1], S_train.shape[-1], 96, 16, N_CLASSES_OUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-3)

    best_f1, best_state, no_imp = 0.0, None, 0
    print(f"{'Epoch':>6} {'Train':>9} {'ValF1':>7}")

    for epoch in range(1, 101):
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

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, sb, yb in val_ld:
                preds.extend(model(xb.to(DEVICE), sb.to(DEVICE)).argmax(1).cpu().numpy())
                trues.extend(yb.numpy())
        vf1 = f1_score(trues, preds, average="macro", zero_division=0)

        if epoch % 10 == 0:
            print(f"{epoch:>6} {tloss:>9.4f} {vf1:>7.4f}")

        if vf1 > best_f1:
            best_f1, best_state, no_imp = vf1, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= 20:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, sb, yb in test_ld:
            preds.extend(model(xb.to(DEVICE), sb.to(DEVICE)).argmax(1).cpu().numpy())
            trues.extend(yb.numpy())
    preds, trues = np.array(preds), np.array(trues)

    return {
        "Accuracy": accuracy_score(trues, preds),
        "Macro F1": f1_score(trues, preds, average="macro", zero_division=0),
        "MAE":      mean_absolute_error(trues, preds),
        "RMSE":     np.sqrt(mean_squared_error(trues, preds)),
    }

# ------------------------------------------------------------------ #
# XGBoost baseline
# ------------------------------------------------------------------ #
def extract_features(X, S):
    mean = X.mean(axis=1)
    std  = X.std(axis=1)
    mn   = X.min(axis=1)
    mx   = X.max(axis=1)
    trend = (X[:, -1, :] - X[:, 0, :]) / X.shape[1]
    return np.concatenate([mean, std, mn, mx, trend, S], axis=1)

def train_xgboost(X_train, S_train, y_train, X_test, S_test, y_test):
    print("\n" + "="*50)
    print("BASELINE 1: XGBoost (flattened statistical + soil), 'None' excluded")
    print("="*50)

    X_tr_flat = extract_features(X_train, S_train)
    X_te_flat = extract_features(X_test, S_test)
    print(f"  Feature vector shape: {X_tr_flat.shape}")

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([cw[c] for c in y_train])

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=SEED, n_jobs=-1, device="cpu",
    )
    print("  Training XGBoost...")
    model.fit(X_tr_flat, y_train, sample_weight=sample_weights, verbose=False)

    preds = model.predict(X_te_flat)
    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Macro F1": f1_score(y_test, preds, average="macro", zero_division=0),
        "MAE":      mean_absolute_error(y_test, preds),
        "RMSE":     np.sqrt(mean_squared_error(y_test, preds)),
    }

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} not found -- run load_data.py first.")
    data = np.load(DATA_PATH, allow_pickle=True)
    X_train_full, y_train_full, S_train_full = data["X_train"], data["y_train"], data["S_train"]
    X_val_full,   y_val_full,   S_val_full   = data["X_val"],   data["y_val"],   data["S_val"]
    X_test_full,  y_test_full,  S_test_full  = data["X_test"],  data["y_test"],  data["S_test"]

    print(f"Full dataset (before dropping None): "
          f"{np.bincount(y_train_full, minlength=CLASS_SCHEME)} ({FULL_LABELS})")

    X_train, S_train, y_train = drop_none_and_remap(X_train_full, S_train_full, y_train_full)
    X_val,   S_val,   y_val   = drop_none_and_remap(X_val_full,   S_val_full,   y_val_full)
    X_test,  S_test,  y_test  = drop_none_and_remap(X_test_full,  S_test_full,  y_test_full)

    print(f"After dropping None ({OUT_LABELS}): "
          f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")
    print(f"Train class dist: {np.bincount(y_train, minlength=N_CLASSES_OUT)}")

    xgb_results  = train_xgboost(X_train, S_train, y_train, X_test, S_test, y_test)
    lstm_results = train_lstm_only(X_train, S_train, y_train, X_val, S_val, y_val, X_test, S_test, y_test)

    save_results("XGBoost (baseline, no-None)", xgb_results,
                 extra={"excludes_none": True})
    save_results("LSTM-only (baseline, no-None)", lstm_results,
                 extra={"excludes_none": True})

    print("\n" + "="*65)
    print("MODEL COMPARISON TABLE ('None' excluded)")
    print("="*65)
    print(f"{'Model':<28} {'Accuracy':>9} {'Macro F1':>9} {'MAE':>8} {'RMSE':>8}")
    print("-"*65)
    for name, res in [("XGBoost (no-None)", xgb_results), ("LSTM-only (no-None)", lstm_results)]:
        print(f"{name:<28} {res['Accuracy']:>9.4f} {res['Macro F1']:>9.4f} "
              f"{res['MAE']:>8.4f} {res['RMSE']:>8.4f}")
    print("(Compare against LSTM+Attention (no-None, severity-only) in results.json,")
    print(" saved separately by train_no_none.py)")
    print("="*65)

if __name__ == "__main__":
    main()
