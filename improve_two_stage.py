import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             confusion_matrix)

RESULTS_PATH = "results.json"
DATA_PATH = "sequences_kazakhstan_features.npz"
N_CLASSES_4 = 4
CLASS_LABELS_4 = ["None", "Mild", "Moderate", "Severe+"]

PROJ_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ = 24, 64, 1, 0.4, 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# seed -> (binary gate weights, severity weights, 4-class rescue weights)
SEED_WEIGHTS = {
    42:   ("model_kazakhstan_binary.pt",
           "model_kazakhstan_severity.pt",
           "model_kazakhstan_features_seed42.pt"),
    123:  ("model_kazakhstan_binary_seed123.pt",
           "model_kazakhstan_severity_seed123.pt",
           "model_kazakhstan_features_seed123.pt"),
    2026: ("model_kazakhstan_binary_seed2026.pt",
           "model_kazakhstan_severity_seed2026.pt",
           "model_kazakhstan_features_seed2026.pt"),
}

# Reference single-stage macro F1 per seed (from the multi-seed runs)
SINGLE_STAGE_F1 = {42: 0.6285, 123: 0.6241, 2026: 0.6319}

# Grids (probability thresholds on the binary gate)
TAU_LO_GRID = [0.05, 0.15, 0.25, 0.35, 0.45]      # confident no-drought below this
TAU_HI_GRID = [0.55, 0.65, 0.75, 0.85, 0.95]      # confident drought above this
TAU_SINGLE_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


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


def load_model(path, n_classes, input_size, soil_size):
    model = LSTMAttention(input_size, soil_size, PROJ_SIZE, HIDDEN_SIZE,
                          N_LAYERS, DROPOUT, SOIL_PROJ, n_classes)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def predict_probs(model, X, S):
    """Return softmax probabilities over classes, as a numpy array."""
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    St = torch.tensor(S, dtype=torch.float32).to(DEVICE)
    probs = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            logits = model(Xt[i:i + bs], St[i:i + bs])
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def macro_f1(trues, preds):
    return f1_score(trues, preds, average="macro", zero_division=0)


def hard_gate_preds(p_bin, sev_pred, tau):
    """No rescue: class 0 below tau, severity+1 above."""
    out = np.zeros(len(p_bin), dtype=np.int64)
    drought = p_bin[:, 1] >= tau
    out[drought] = sev_pred[drought] + 1
    return out


def rescue_preds(p_bin, sev_pred, rescue_pred, tau_lo, tau_hi):
    """Three-way routing with rescue model in the ambiguous band."""
    out = np.zeros(len(p_bin), dtype=np.int64)
    pd = p_bin[:, 1]
    confident_none = pd <= tau_lo
    confident_drought = pd >= tau_hi
    ambiguous = ~(confident_none | confident_drought)
    out[confident_none] = 0
    out[confident_drought] = sev_pred[confident_drought] + 1
    out[ambiguous] = rescue_pred[ambiguous]
    return out


def full_metrics(trues, preds):
    acc = accuracy_score(trues, preds)
    f1 = macro_f1(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    return acc, f1, mae, rmse


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} not found.")

    data = np.load(DATA_PATH, allow_pickle=True)
    X_train, y_train, S_train = data["X_train"], data["y_train"], data["S_train"]
    X_val, y_val, S_val = data["X_val"], data["y_val"], data["S_val"]
    X_test, y_test, S_test = data["X_test"], data["y_test"], data["S_test"]
    input_size = X_train.shape[-1]
    soil_size = S_train.shape[-1]

    S_dummy_val = np.zeros((len(X_val), soil_size), dtype=np.float32)
    S_dummy_test = np.zeros((len(X_test), soil_size), dtype=np.float32)

    summary_rows = []
    for seed, (bin_path, sev_path, rescue_path) in SEED_WEIGHTS.items():
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        model_bin = load_model(bin_path, 2, input_size, soil_size)
        model_sev = load_model(sev_path, 3, input_size, soil_size)
        model_res = load_model(rescue_path, 4, input_size, soil_size)

        # ---- precompute per-dataset predictions once ----
        va_p_bin = predict_probs(model_bin, X_val, S_dummy_val)
        va_sev = predict_probs(model_sev, X_val, S_dummy_val).argmax(1)
        va_res = predict_probs(model_res, X_val, S_dummy_val).argmax(1)

        te_p_bin = predict_probs(model_bin, X_test, S_dummy_test)
        te_sev = predict_probs(model_sev, X_test, S_dummy_test).argmax(1)
        te_res = predict_probs(model_res, X_test, S_dummy_test).argmax(1)

        # ---- control: hard gate at 0.5 (must reproduce 0.227) ----
        acc_c, f1_c, mae_c, rmse_c = full_metrics(y_test, hard_gate_preds(te_p_bin, te_sev, 0.5))
        print(f"  [control] hard gate @0.5        -> Macro F1 {f1_c:.4f}  Acc {acc_c:.4f}  "
              f"(expect ~0.227)")

        ctrl_extra = {"seed": seed, "stage": "two-stage-combined",
                      "gate_threshold": 0.5, "corrected_assembly": True,
                      "note": "batch-index bug fixed in train_two_stage.py combined eval"}
        save_results(f"LSTM+Attention (KZ, two-stage combined, seed{seed})",
                     {"Accuracy": acc_c, "Macro F1": f1_c, "MAE": mae_c, "RMSE": rmse_c},
                     extra=ctrl_extra)
        if seed == 42:
            save_results("LSTM+Attention (KZ, two-stage combined)",
                         {"Accuracy": acc_c, "Macro F1": f1_c, "MAE": mae_c, "RMSE": rmse_c},
                         extra=ctrl_extra)

        # ---- Variant A: single-threshold tuned gate (no rescue) ----
        best_a = None
        for tau in TAU_SINGLE_GRID:
            f1v = macro_f1(y_val, hard_gate_preds(va_p_bin, va_sev, tau))
            if best_a is None or f1v > best_a[1]:
                best_a = (tau, f1v)
        tau_a = best_a[0]
        acc_a, f1_a, mae_a, rmse_a = full_metrics(y_test, hard_gate_preds(te_p_bin, te_sev, tau_a))
        print(f"  [A] tuned gate      tau={tau_a:<4} -> Macro F1 {f1_a:.4f}  Acc {acc_a:.4f}")

        # ---- Variant C: three-way rescue routing (best on validation) ----
        best_c = None
        for tau_lo in TAU_LO_GRID:
            for tau_hi in TAU_HI_GRID:
                f1v = macro_f1(y_val, rescue_preds(va_p_bin, va_sev, va_res, tau_lo, tau_hi))
                if best_c is None or f1v > best_c[2]:
                    best_c = (tau_lo, tau_hi, f1v)
        tl, th = best_c[0], best_c[1]
        preds_c = rescue_preds(te_p_bin, te_sev, te_res, tl, th)
        acc_c2, f1_c2, mae_c2, rmse_c2 = full_metrics(y_test, preds_c)
        print(f"  [C] rescue routing  lo={tl:<4} hi={th:<4} -> Macro F1 {f1_c2:.4f}  Acc {acc_c2:.4f}")

        cm = confusion_matrix(y_test, preds_c, labels=list(range(N_CLASSES_4)))
        print(f"  Confusion Matrix (C):\n{cm}")

        bucket = {}
        pd_te = te_p_bin[:, 1]
        bucket["none"] = int((pd_te <= tl).sum())
        bucket["drought"] = int((pd_te >= th).sum())
        bucket["rescue"] = int(((pd_te > tl) & (pd_te < th)).sum())
        print(f"  Routing split on test: {bucket}")

        save_results(f"LSTM+Attention (KZ, two-stage tuned gate, seed{seed})",
                     {"Accuracy": acc_a, "Macro F1": f1_a, "MAE": mae_a, "RMSE": rmse_a},
                     extra={"seed": seed, "stage": "two-stage-tuned-gate",
                            "tau": tau_a, "n_features": input_size})
        save_results(f"LSTM+Attention (KZ, two-stage + rescue routing, seed{seed})",
                     {"Accuracy": acc_c2, "Macro F1": f1_c2, "MAE": mae_c2, "RMSE": rmse_c2},
                     extra={"seed": seed, "stage": "two-stage-rescue-routing",
                            "tau_lo": tl, "tau_hi": th, "routing_test_split": bucket,
                            "n_features": input_size})

        summary_rows.append((seed, f1_c, f1_a, f1_c2,
                             SINGLE_STAGE_F1[seed], tau_a, tl, th))

    print(f"\n{'='*70}\nFINAL COMPARISON (Macro F1)\n{'='*70}")
    print(f"{'Seed':<6}{'control':>9}{'tuned-gate':>11}{'rescue':>8}{'single':>9}"
          f"{'tauA':>6}{'lo':>5}{'hi':>5}")
    for (s, c, a, r, ss, ta, tl, th) in summary_rows:
        print(f"{s:<6}{c:>9.4f}{a:>11.4f}{r:>8.4f}{ss:>9.4f}{ta:>6.2f}{tl:>5.2f}{th:>5.2f}")
    print(f"\nSingle-stage remains the reference; rescue routing should recover most "
          f"of the -0.40 gap but is not expected to pass single-stage.")


if __name__ == "__main__":
    main()