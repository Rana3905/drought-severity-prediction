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

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_SCHEME = 4       # <-- must match load_data.py; loads sequences_{N}.npz
N_CLASSES  = CLASS_SCHEME
DATA_PATH  = f"sequences_{CLASS_SCHEME}.npz"
BATCH_SIZE = 512
SEED       = 42

RESULTS_PATH = "results.json"

def save_results(model_name, metrics, extra=None):
    """Append this run's results to results.json under model_name (creates
    file if missing). Each model_name maps to a LIST of runs, so past runs
    (e.g. different N_CLASSES or county counts) are never overwritten."""
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
    print(f"Saved run #{len(existing)} for '{model_name}' (n_classes={N_CLASSES}) -> {RESULTS_PATH}")

def load_latest_result(model_name, all_results=None, n_classes=None):
    """Return the most recent run's metrics for model_name, or None.
    If n_classes is given, only considers runs matching that class scheme."""
    if all_results is None:
        all_results = load_results()
    entries = all_results.get(model_name)
    if not entries:
        return None
    if isinstance(entries, dict):
        entries = [entries]
    if n_classes is not None:
        entries = [e for e in entries if e.get("n_classes") == n_classes]
        if not entries:
            return None
    return entries[-1]  # most recent matching run

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

# ------------------------------------------------------------------ #
# LSTM+Attention results -- fallback only, used if results.json has no
# entry yet (i.e. train_model.py hasn't been run in this folder).
# Real values are read from results.json, saved automatically by
# train_model.py after each run.
# ------------------------------------------------------------------ #
LSTM_ATTN_FALLBACK = {
    "Accuracy": 0.7561,
    "Macro F1": 0.4586,
    "MAE":      0.3512,
    "RMSE":     0.7873,
}

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
# LSTM-only model (no attention, but keeps the soil branch for parity)
# ------------------------------------------------------------------ #
class LSTMOnly(nn.Module):
    def __init__(self, input_size, soil_size, hidden_size, soil_proj, n_classes, dropout=0.4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LayerNorm(32), nn.ReLU(), nn.Dropout(dropout*0.5),
        )
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.soil_proj = nn.Sequential(
            nn.Linear(soil_size, soil_proj),
            nn.LayerNorm(soil_proj), nn.ReLU(), nn.Dropout(dropout*0.5),
        )
        self.fc1  = nn.Linear(hidden_size + soil_proj, 32)
        self.fc2  = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x, s):
        B, T, F = x.shape
        x = self.proj(x.reshape(B*T, F)).reshape(B, T, -1)
        _, (h, _) = self.lstm(x)          # use final hidden state only
        soil_ctx = self.soil_proj(s)
        combined = torch.cat([self.drop(h[-1]), soil_ctx], dim=1)
        out = self.fc2(self.relu(self.fc1(combined)))
        return out

def train_lstm_only(X_train, S_train, y_train, X_val, S_val, y_val,
                     X_test, S_test, y_test):
    print("\n" + "="*50)
    print("BASELINE 2: LSTM-only (no attention)")
    print("="*50)

    train_ld = DataLoader(DroughtDataset(X_train, S_train, y_train), BATCH_SIZE, shuffle=True)
    val_ld   = DataLoader(DroughtDataset(X_val, S_val, y_val),   BATCH_SIZE*2)
    test_ld  = DataLoader(DroughtDataset(X_test, S_test, y_test),  BATCH_SIZE*2)

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = np.clip(np.sqrt(cw), None, 2.0).astype(np.float32)
    cw_t = torch.tensor(cw).to(DEVICE)

    model     = LSTMOnly(X_train.shape[-1], S_train.shape[-1], 96, 16, N_CLASSES).to(DEVICE)
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
    """Flatten (N, 90, 18) -> tabular stats, then append static soil vector."""
    mean = X.mean(axis=1)
    std  = X.std(axis=1)
    mn   = X.min(axis=1)
    mx   = X.max(axis=1)
    # Linear trend slope per feature (last - first) / seq_len
    trend = (X[:, -1, :] - X[:, 0, :]) / X.shape[1]
    return np.concatenate([mean, std, mn, mx, trend, S], axis=1)

def train_xgboost(X_train, S_train, y_train, X_test, S_test, y_test):
    print("\n" + "="*50)
    print("BASELINE 1: XGBoost (flattened statistical + soil features)")
    print("="*50)

    X_tr_flat = extract_features(X_train, S_train)
    X_te_flat = extract_features(X_test, S_test)
    print(f"  Feature vector shape: {X_tr_flat.shape}")

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([cw[c] for c in y_train])

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
        device="cpu",
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
    data    = np.load(DATA_PATH, allow_pickle=True)
    X_train = data["X_train"]; y_train = data["y_train"]; S_train = data["S_train"]
    X_val   = data["X_val"];   y_val   = data["y_val"];   S_val   = data["S_val"]
    X_test  = data["X_test"];  y_test  = data["y_test"];  S_test  = data["S_test"]

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    xgb_results  = train_xgboost(X_train, S_train, y_train, X_test, S_test, y_test)
    lstm_results = train_lstm_only(X_train, S_train, y_train, X_val, S_val, y_val,
                                    X_test, S_test, y_test)

    save_results("XGBoost (baseline)", xgb_results)
    save_results("LSTM-only (baseline)", lstm_results)

    # Pull the most recent LSTM+Attention run if train_model.py has already
    # saved one to results.json; otherwise fall back to the last known
    # numbers so this script still runs standalone.
    saved = load_results()
    lstm_attn_results = load_latest_result("LSTM+Attention (ours)", saved, n_classes=CLASS_SCHEME) or LSTM_ATTN_FALLBACK
    if load_latest_result("LSTM+Attention (ours)", saved, n_classes=CLASS_SCHEME) is None:
        print(f"\n[Note] No LSTM+Attention entry found in results.json for "
              f"CLASS_SCHEME={CLASS_SCHEME} -- using fallback numbers. "
              f"Run train_model.py with matching CLASS_SCHEME first for live results.")

    # ---------------------------------------------------------------- #
    # Final comparison table
    # ---------------------------------------------------------------- #
    print("\n" + "="*65)
    print("MODEL COMPARISON TABLE")
    print("="*65)
    print(f"{'Model':<25} {'Accuracy':>9} {'Macro F1':>9} {'MAE':>8} {'RMSE':>8}")
    print("-"*65)

    models = {
        "XGBoost (baseline)":      xgb_results,
        "LSTM-only (baseline)":    lstm_results,
        "LSTM+Attention (ours)":   lstm_attn_results,
    }

    for name, res in models.items():
        print(f"{name:<25} {res['Accuracy']:>9.4f} {res['Macro F1']:>9.4f} "
              f"{res['MAE']:>8.4f} {res['RMSE']:>8.4f}")
    print("="*65)

if __name__ == "__main__":
    main()
