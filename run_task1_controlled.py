"""Task 1: controlled comparison of Scenario C vs Scenario D on shared architecture.

Uses the frozen shared model (`LSTMAttention`, `make_zero_static` from
model_lstm_attention.py) and the shared metrics (metrics_task_spec.py).
Training recipe mirrors train_model_kazakhstan_features.py: class-weighted CE
(sqrt-scaled, capped 2.0), AdamW 5e-4 / 5e-3, 5-epoch warmup + cosine to 0,
grad clip 1.0, label smoothing 0.1, 150 epochs / patience 30 on val Macro F1.

Writes task1_results.json with keys {"individual_runs", "summary_by_scenario"},
saving after EVERY (scenario, seed) run so a crash never loses prior results.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader

from metrics_task_spec import multiclass_metrics, summarize_runs
from model_lstm_attention import LSTMAttention

RESULT_PATH = "task1_results.json"

SCENARIOS_TO_RUN = list(sys.argv[1:]) or ["C"]

SEEDS = [42, 123, 2026]

BATCH_SIZE, LR, WEIGHT_DECAY, LABEL_SMOOTH = 64, 5e-4, 5e-3, 0.1
MAX_EPOCHS, PATIENCE, MAX_CW, WARMUP = 150, 30, 2.0, 5
N_CLASSES = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS = ["None", "Mild", "Moderate", "Severe+"]
SUMMARY_METRICS = ["Accuracy", "Macro F1", "MAE", "RMSE", "Severe+ False Negative Rate"]

SCENARIO_CONFIGS = {
    "C": {
        "title": "Scenario C: raw 6-month SPEI, 1 feature",
        "data_path": "sequences_task1_kz_raw.npz",
    },
    "D": {
        "title": "Scenario D: engineered 6-month SPEI features, 22 features",
        "data_path": "sequences_task1_kz_features.npz",
    },
}


class DroughtDataset(Dataset):
    def __init__(self, X, S, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.S[i], self.y[i]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_class_weights(y_train):
    cw_vals = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = np.clip(np.sqrt(cw_vals), None, MAX_CW).astype(np.float32)
    return torch.tensor(cw, dtype=torch.float32)


def train_eval_scenario(cfg, seed):
    data = np.load(cfg["data_path"], allow_pickle=True)
    X_train, y_train, S_train = data["X_train"], data["y_train"], data["S_train"]
    X_val, y_val, S_val = data["X_val"], data["y_val"], data["S_val"]
    X_test, y_test, S_test = data["X_test"], data["y_test"], data["S_test"]
    feature_names = list(data["feature_names"])

    print(f"\n{'=' * 70}\n{cfg['title']} | seed={seed}\n{'=' * 70}")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Train class dist: {np.bincount(y_train, minlength=N_CLASSES)} ({CLASS_LABELS})")
    if S_train.shape[1] != 31:
        raise SystemExit(f"Unexpected soil dim {S_train.shape[1]} (expected 31).")

    train_ld = DataLoader(DroughtDataset(X_train, S_train, y_train), BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(DroughtDataset(X_val, S_val, y_val), BATCH_SIZE * 2)
    test_ld = DataLoader(DroughtDataset(X_test, S_test, y_test), BATCH_SIZE * 2)

    set_seed(seed)  # INSIDE per-seed loop, before model creation
    model = LSTMAttention(input_size=X_train.shape[-1], n_classes=N_CLASSES).to(DEVICE)

    cw = make_class_weights(y_train).to(DEVICE)
    print(f"Class weights: {cw.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP:
            return epoch / WARMUP
        progress = (epoch - WARMUP) / (MAX_EPOCHS - WARMUP)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, best_state, no_imp, epochs_run = 0.0, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, sb, yb in train_ld:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb, sb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        epochs_run = epoch

        vpred, vtrue = [], []
        model.eval()
        with torch.no_grad():
            for xb, sb, yb in val_ld:
                xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
                vpred.extend(model(xb, sb).argmax(1).cpu().numpy())
                vtrue.extend(yb.cpu().numpy())
        vf1v = multiclass_metrics(np.array(vtrue), np.array(vpred))["Macro F1"]

        if vf1v > best_f1:
            best_f1, best_state, no_imp = (
                vf1v,
                {k: v.cpu().clone() for k, v in model.state_dict().items()},
                0,
            )
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    print(f"Trained {epochs_run} epochs; best val Macro F1: {best_f1:.4f}")

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, sb, yb in test_ld:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            preds.extend(model(xb, sb).argmax(1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    preds = np.array(preds)
    trues = np.array(trues)
    assert len(preds) == len(y_test)

    metrics = multiclass_metrics(trues, preds)
    print(f"seed={seed} | Accuracy: {metrics['Accuracy']:.4f} | "
          f"Macro F1: {metrics['Macro F1']:.4f} | MAE: {metrics['MAE']:.4f} | "
          f"RMSE: {metrics['RMSE']:.4f}")

    return {
        "scenario": cfg["scenario"],
        "seed": seed,
        "epochs": epochs_run,
        "best_val_macro_f1": best_f1,
        "n_features": int(X_train.shape[-1]),
        "feature_names": feature_names,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }


def load_results():
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_results(individual_runs):
    summary_by_scenario = {
        sc: summarize_runs(rows, SUMMARY_METRICS)
        for sc, rows in individual_runs.items()
    }
    out = {
        "individual_runs": individual_runs,
        "summary_by_scenario": summary_by_scenario,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {RESULT_PATH}")


def main():
    print(f"Device: {DEVICE} | Seeds: {SEEDS}")
    print(f"Running scenarios: {SCENARIOS_TO_RUN}")

    all_runs = load_results().get("individual_runs", {})
    for sc in SCENARIOS_TO_RUN:
        if sc not in SCENARIO_CONFIGS:
            raise SystemExit(f"Unknown scenario '{sc}'.")
        cfg = {**SCENARIO_CONFIGS[sc], "scenario": sc}

        if not os.path.exists(cfg["data_path"]):
            raise SystemExit(f"{cfg['data_path']} not found -- run the loader first.")

        scenario_runs = all_runs.get(sc, [])
        for seed in SEEDS:
            try:
                entry = train_eval_scenario(cfg, seed)
            except Exception as exc:
                entry = {
                    "scenario": sc,
                    "seed": seed,
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                print(f"\n[ERROR] scenario {sc} seed {seed} failed: {exc}")
            scenario_runs = [r for r in scenario_runs if r.get("seed") != seed]
            scenario_runs.append(entry)
            all_runs[sc] = sorted(scenario_runs, key=lambda r: r.get("seed", 0))
            save_results(all_runs)

    print("\n" + "=" * 70)
    print("TASK 1 SUMMARY (Macro F1, mean +/- std over seeds)")
    print("=" * 70)
    summary = load_results().get("summary_by_scenario", {})
    for sc in ("C", "D"):
        if sc not in summary:
            continue
        m = summary[sc].get("Macro F1", {})
        a = summary[sc].get("Accuracy", {})
        print(f"  Scenario {sc}: Macro F1 {m.get('mean', float('nan')):.4f} "
              f"+/- {m.get('std', 0.0):.4f} | Accuracy {a.get('mean', float('nan')):.4f}")
    if "C" in summary and "D" in summary:
        c, d = summary["C"]["Macro F1"], summary["D"]["Macro F1"]
        print(f"  C -> D Macro F1 delta: {d['mean'] - c['mean']:+.4f}")


if __name__ == "__main__":
    main()