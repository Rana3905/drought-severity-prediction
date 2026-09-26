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
N_CLASSES_4 = 4
N_CLASSES_BINARY = 2
N_CLASSES_SEVERITY = 3
CLASS_LABELS_4 = ["None", "Mild", "Moderate", "Severe+"]
CLASS_LABELS_BINARY = ["No Drought", "Drought"]
CLASS_LABELS_SEVERITY = ["Mild", "Moderate", "Severe+"]

PROJ_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ = 24, 64, 1, 0.4, 8
BATCH_SIZE, LR, WEIGHT_DECAY, LABEL_SMOOTH = 64, 5e-4, 5e-3, 0.1
MAX_EPOCHS, PATIENCE, MAX_CW = 150, 30, 2.0

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


def train_model(model, train_ld, val_ld, criterion, max_epochs, patience, tag):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (max_epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, best_state, no_imp = 0.0, None, 0
    for epoch in range(1, max_epochs + 1):
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
            print(f"    {tag} Epoch {epoch:>3}: train={tloss:.4f} val={vloss:.4f} "
                  f"valF1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1, best_state, no_imp = vf1, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"    {tag} early stopping at epoch {epoch} (best val F1: {best_f1:.4f})")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model


def main():
    SEED = int(os.environ.get("SEED", 42))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} not found -- run load_data_kazakhstan_features.py first.")

    data = np.load(DATA_PATH, allow_pickle=True)
    X_train, y_train, S_train = data["X_train"], data["y_train"], data["S_train"]
    X_val, y_val, S_val = data["X_val"], data["y_val"], data["S_val"]
    X_test, y_test, S_test = data["X_test"], data["y_test"], data["S_test"]
    feature_names = data["feature_names"]

    print(f"Device: {DEVICE} | SEED: {SEED}")
    print(f"Data: {list(feature_names)} ({X_train.shape[-1]} features)")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Train class dist: {np.bincount(y_train, minlength=N_CLASSES_4)}")

    S_dummy_val = np.zeros((len(X_val), 31), dtype=np.float32)
    S_dummy_test = np.zeros((len(X_test), 31), dtype=np.float32)

    # =========================================
    # STAGE 0: Single-stage 4-class baseline
    # =========================================
    print(f"\n{'='*60}")
    print(f"STAGE 0: Single-stage 4-class baseline")
    print(f"{'='*60}")

    cw4 = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw4 = np.clip(np.sqrt(cw4), None, MAX_CW).astype(np.float32)
    cw4_t = torch.tensor(cw4).to(DEVICE)
    crit4 = nn.CrossEntropyLoss(weight=cw4_t, label_smoothing=LABEL_SMOOTH)

    train_ld = DataLoader(DroughtDataset(X_train, S_train, y_train),
                          BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(DroughtDataset(X_val, S_dummy_val, y_val), BATCH_SIZE * 2)
    test_ld = DataLoader(DroughtDataset(X_test, S_dummy_test, y_test), BATCH_SIZE * 2)

    model_single = LSTMAttention(X_train.shape[-1], S_train.shape[-1], PROJ_SIZE,
                                 HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ,
                                 N_CLASSES_4).to(DEVICE)
    model_single = train_model(model_single, train_ld, val_ld, crit4,
                               MAX_EPOCHS, PATIENCE, "4-cls")
    torch.save(model_single.state_dict(), f"model_kazakhstan_two_stage_base_seed{SEED}.pt")

    _, acc4, f1_4, mae4, rmse4, preds4, trues4 = evaluate(model_single, test_ld, crit4)
    cm4 = confusion_matrix(trues4, preds4, labels=list(range(N_CLASSES_4)))
    print(f"  Single-stage 4-class: Accuracy={acc4:.4f}, Macro F1={f1_4:.4f}")
    print(f"  Confusion Matrix:\n{cm4}")

    save_results(f"LSTM+Attention (KZ, two-stage base, 4-class, seed{SEED})",
                 {"Accuracy": acc4, "Macro F1": f1_4, "MAE": mae4, "RMSE": rmse4},
                 extra={"n_features": int(X_train.shape[-1]), "seed": SEED,
                        "stage": "single-stage-baseline"})

    # =========================================
    # STAGE 1: Binary drought detector
    # =========================================
    print(f"\n{'='*60}")
    print(f"STAGE 1: Binary drought/no-drought classifier")
    print(f"{'='*60}")

    y_train_bin = (y_train > 0).astype(np.int64)
    y_val_bin = (y_val > 0).astype(np.int64)
    y_test_bin = (y_test > 0).astype(np.int64)
    print(f"  Binary train dist: {np.bincount(y_train_bin, minlength=2)}")

    cw_bin = compute_class_weight("balanced", classes=np.unique(y_train_bin), y=y_train_bin)
    cw_bin = np.clip(np.sqrt(cw_bin), None, MAX_CW).astype(np.float32)
    cw_bin_t = torch.tensor(cw_bin).to(DEVICE)
    crit_bin = nn.CrossEntropyLoss(weight=cw_bin_t, label_smoothing=LABEL_SMOOTH)

    train_ld_bin = DataLoader(DroughtDataset(X_train, S_train, y_train_bin),
                              BATCH_SIZE, shuffle=True)
    val_ld_bin = DataLoader(DroughtDataset(X_val, S_dummy_val, y_val_bin), BATCH_SIZE * 2)
    test_ld_bin = DataLoader(DroughtDataset(X_test, S_dummy_test, y_test_bin), BATCH_SIZE * 2)

    model_bin = LSTMAttention(X_train.shape[-1], S_train.shape[-1], PROJ_SIZE,
                              HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ,
                              N_CLASSES_BINARY).to(DEVICE)
    model_bin = train_model(model_bin, train_ld_bin, val_ld_bin, crit_bin,
                            MAX_EPOCHS, PATIENCE, "binary")
    torch.save(model_bin.state_dict(), f"model_kazakhstan_binary_seed{SEED}.pt")

    _, acc_bin, f1_bin, mae_bin, rmse_bin, preds_bin, trues_bin = evaluate(
        model_bin, test_ld_bin, crit_bin)
    cm_bin = confusion_matrix(trues_bin, preds_bin, labels=list(range(N_CLASSES_BINARY)))
    print(f"  Binary: Accuracy={acc_bin:.4f}, Macro F1={f1_bin:.4f}")
    print(f"  Confusion Matrix (0=NoDrought, 1=Drought):\n{cm_bin}")

    save_results(f"LSTM+Attention (KZ, binary stage 1, seed{SEED})",
                 {"Accuracy": acc_bin, "Macro F1": f1_bin, "MAE": mae_bin, "RMSE": rmse_bin},
                 extra={"n_features": int(X_train.shape[-1]), "seed": SEED,
                        "stage": "binary-detector"})

    # =========================================
    # STAGE 2: Severity classifier (on drought samples only)
    # =========================================
    print(f"\n{'='*60}")
    print(f"STAGE 2: Severity classifier (Mild/Moderate/Severe+)")
    print(f"{'='*60}")

    drought_mask_train = y_train > 0
    drought_mask_val = y_val > 0
    drought_mask_test = y_test > 0

    X_tr_sev = X_train[drought_mask_train]
    S_tr_sev = S_train[drought_mask_train]
    y_tr_sev = (y_train[drought_mask_train] - 1).astype(np.int64)

    X_va_sev = X_val[drought_mask_val]
    S_va_sev = S_dummy_val[drought_mask_val]
    y_va_sev = (y_val[drought_mask_val] - 1).astype(np.int64)

    X_te_sev = X_test[drought_mask_test]
    S_te_sev = S_dummy_test[drought_mask_test]
    y_te_sev = (y_test[drought_mask_test] - 1).astype(np.int64)

    print(f"  Severity train dist: {np.bincount(y_tr_sev, minlength=N_CLASSES_SEVERITY)}")

    cw_sev = compute_class_weight("balanced", classes=np.unique(y_tr_sev), y=y_tr_sev)
    cw_sev = np.clip(np.sqrt(cw_sev), None, MAX_CW).astype(np.float32)
    cw_sev_t = torch.tensor(cw_sev).to(DEVICE)
    crit_sev = nn.CrossEntropyLoss(weight=cw_sev_t, label_smoothing=LABEL_SMOOTH)

    train_ld_sev = DataLoader(DroughtDataset(X_tr_sev, S_tr_sev, y_tr_sev),
                              BATCH_SIZE, shuffle=True)
    val_ld_sev = DataLoader(DroughtDataset(X_va_sev, S_va_sev, y_va_sev), BATCH_SIZE * 2)
    test_ld_sev = DataLoader(DroughtDataset(X_te_sev, S_te_sev, y_te_sev), BATCH_SIZE * 2)

    model_sev = LSTMAttention(X_train.shape[-1], S_train.shape[-1], PROJ_SIZE,
                              HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ,
                              N_CLASSES_SEVERITY).to(DEVICE)
    model_sev = train_model(model_sev, train_ld_sev, val_ld_sev, crit_sev,
                            MAX_EPOCHS, PATIENCE, "sev")
    torch.save(model_sev.state_dict(), f"model_kazakhstan_severity_seed{SEED}.pt")

    _, acc_sev, f1_sev, mae_sev, rmse_sev, preds_sev, trues_sev = evaluate(
        model_sev, test_ld_sev, crit_sev)
    cm_sev = confusion_matrix(trues_sev, preds_sev, labels=list(range(N_CLASSES_SEVERITY)))
    print(f"  Severity: Accuracy={acc_sev:.4f}, Macro F1={f1_sev:.4f}")
    print(f"  Confusion Matrix:\n{cm_sev}")

    save_results(f"LSTM+Attention (KZ, severity stage 2, seed{SEED})",
                 {"Accuracy": acc_sev, "Macro F1": f1_sev, "MAE": mae_sev, "RMSE": rmse_sev},
                 extra={"n_features": int(X_train.shape[-1]), "seed": SEED,
                        "stage": "severity-classifier"})

    # =========================================
    # Two-stage combined evaluation
    # =========================================
    print(f"\n{'='*60}")
    print(f"TWO-STAGE COMBINED PREDICTION")
    print(f"{'='*60}")

    model_bin.eval()
    model_sev.eval()

    all_preds_2stage = []
    with torch.no_grad():
        X_te_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        S_te_t = torch.tensor(S_dummy_test, dtype=torch.float32).to(DEVICE)
        bs = 256
        for i in range(0, len(X_te_t), bs):
            xb = X_te_t[i:i+bs]
            sb = S_te_t[i:i+bs]
            bin_logits = model_bin(xb, sb)
            bin_preds = bin_logits.argmax(1).cpu().numpy()

            sev_indices = np.where(bin_preds == 1)[0]
            if len(sev_indices) > 0:
                xb_sev = xb[sev_indices]
                sb_sev = sb[sev_indices]
                sev_logits = model_sev(xb_sev, sb_sev)
                sev_preds = sev_logits.argmax(1).cpu().numpy() + 1

                for j, idx in enumerate(sev_indices):
                    all_preds_2stage.append((int(i + idx), int(sev_preds[j])))
            for idx in np.where(bin_preds == 0)[0]:
                all_preds_2stage.append((int(i + idx), 0))

    preds_2stage = np.zeros(len(X_test), dtype=np.int64)
    for idx, pred in all_preds_2stage:
        preds_2stage[idx] = pred

    acc_2s = accuracy_score(y_test, preds_2stage)
    f1_2s = f1_score(y_test, preds_2stage, average="macro", zero_division=0)
    mae_2s = mean_absolute_error(y_test, preds_2stage)
    rmse_2s = np.sqrt(mean_squared_error(y_test, preds_2stage))
    cm_2s = confusion_matrix(y_test, preds_2stage, labels=list(range(N_CLASSES_4)))

    print(f"  Two-stage Accuracy : {acc_2s:.4f}")
    print(f"  Two-stage Macro F1 : {f1_2s:.4f}")
    print(f"  Two-stage MAE      : {mae_2s:.4f}")
    print(f"  Two-stage RMSE     : {rmse_2s:.4f}")
    print(f"  Confusion Matrix:\n{cm_2s}")

    save_results(f"LSTM+Attention (KZ, two-stage combined, seed{SEED})",
                 {"Accuracy": acc_2s, "Macro F1": f1_2s, "MAE": mae_2s, "RMSE": rmse_2s},
                 extra={"n_features": int(X_train.shape[-1]), "seed": SEED,
                        "stage": "two-stage-combined",
                        "binary_acc": acc_bin, "binary_f1": f1_bin,
                        "severity_acc": acc_sev, "severity_f1": f1_sev})

    # --- Final Comparison ---
    print(f"\n{'='*72}")
    print(f"TASK 3: TWO-STAGE vs SINGLE-STAGE COMPARISON")
    print(f"{'='*72}")
    print(f"{'Approach':<45} {'Macro F1':>9} {'Accuracy':>9}")
    print(f"{'-'*45} {'-'*9} {'-'*9}")
    print(f"{'Single-stage 4-class':<45} {f1_4:>9.4f} {acc4:>9.4f}")
    print(f"{'Two-stage combined':<45} {f1_2s:>9.4f} {acc_2s:>9.4f}")
    print(f"{'-'*45} {'-'*9} {'-'*9}")
    delta_f1 = f1_2s - f1_4
    delta_acc = acc_2s - acc4
    print(f"{'Difference (2stage - 1stage)':<45} {delta_f1:>+9.4f} {delta_acc:>+9.4f}")
    print(f"\n  Binary detector: Accuracy={acc_bin:.4f}, F1={f1_bin:.4f}")
    print(f"  Severity classifier: Accuracy={acc_sev:.4f}, F1={f1_sev:.4f}")


if __name__ == "__main__":
    main()
