"""Task 2: binary-detector + severity two-stage pipeline vs a single 4-class
baseline on the Scenario D data (seq_len=6, 22 features).

Stage 1: None vs any-drought  (y_binary = y > 0)        -- 2 classes
Stage 2: severity among droughts (y_severity = y - 1)   -- 3 classes
Baseline: single 4-class model on the same split.

Stage 2 is TRAINED on true drought samples only, but at test time it runs on
the samples that stage 1 PREDICTED as drought (predicted-drought routing).
Combined output is assembled with boolean masks over the FULL test array --
never batch-local concatenation; the routing decision is stage-1 predictions,
NOT ground-truth labels (an oracle gate).

Saves task2_results.json.
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader

from metrics_task_spec import (
    FOUR_CLASS_LABELS,
    SEVERITY_LABELS,
    BINARY_LABELS,
    binary_detection_metrics,
    multiclass_metrics,
    task2_critical_errors,
    summarize_runs,
)
from model_lstm_attention import LSTMAttention

RESULT_PATH = "task2_results.json"
DATA_PATH = "sequences_task1_kz_features.npz"

SEEDS = [42, 123, 2026]

BATCH_SIZE, LR, WEIGHT_DECAY, LABEL_SMOOTH = 64, 5e-4, 5e-3, 0.1
MAX_EPOCHS, PATIENCE, MAX_CW, WARMUP = 150, 30, 2.0, 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUMMARY_METRICS = ["Accuracy", "Macro F1", "MAE", "RMSE"]
CRITICAL_ERROR_KEYS = [
    "Severe+ predicted as None rate",
    "Moderate predicted as Mild rate",
    "Mild predicted as Severe+ rate",
    "None predicted as Drought rate",
]


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


def make_class_weights(y):
    cw_vals = compute_class_weight("balanced", classes=np.unique(y), y=y)
    cw = np.clip(np.sqrt(cw_vals), None, MAX_CW).astype(np.float32)
    return torch.tensor(cw, dtype=torch.float32)


@torch.no_grad()
def predict(model, X, S):
    model.eval()
    preds = []
    ld = DataLoader(DroughtDataset(X, S, np.zeros(len(X), dtype=np.int64)), BATCH_SIZE * 2)
    for xb, sb, _ in ld:
        xb, sb = xb.to(DEVICE), sb.to(DEVICE)
        preds.extend(model(xb, sb).argmax(1).cpu().numpy())
    return np.array(preds)


def train_classifier(X_train, y_train, S_train, X_val, y_val, S_val,
                     n_classes, tag, seed, labels=None):
    set_seed(seed)
    train_ld = DataLoader(DroughtDataset(X_train, S_train, y_train), BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(DroughtDataset(X_val, S_val, y_val), BATCH_SIZE * 2)

    model = LSTMAttention(input_size=X_train.shape[-1], n_classes=n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=make_class_weights(y_train).to(DEVICE), label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP:
            return epoch / WARMUP
        progress = (epoch - WARMUP) / (MAX_EPOCHS - WARMUP)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, best_state, no_imp = 0.0, None, 0
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

        vpred = predict(model, X_val, S_val)
        vf1 = multiclass_metrics(y_val, vpred, labels=labels)["Macro F1"]

        if vf1 > best_f1:
            best_f1, best_state, no_imp = (
                vf1,
                {k: v.cpu().clone() for k, v in model.state_dict().items()},
                0,
            )
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model


def assemble_combined(y_test, p_binary_full, p_severity_flagged):
    """Route by STAGE-1 PREDICTIONS (never ground truth), full-array masks."""
    flagged = p_binary_full == 1

    combined = np.zeros(len(y_test), dtype=np.int64)
    combined[flagged] = p_severity_flagged + 1

    assert len(combined) == len(y_test)
    assert len(p_severity_flagged) == int(flagged.sum()), \
        "severity predictions must align to flagged indices"
    assert int(combined[flagged].max() <= 3) == 1
    assert int((combined[~flagged] != 0).sum()) == 0, "non-flagged samples must stay None"
    assert int((combined[flagged] == 0).sum()) == 0, "flagged samples must be routed to severity"

    nz = int((combined != 0).sum())
    assert nz == int(flagged.sum()), f"non-zero count {nz} != flagged {int(flagged.sum())}"

    drought_pos_map = np.cumsum(flagged) - 1
    rng = np.random.default_rng(0)
    spot = rng.choice(len(y_test), size=min(5, len(y_test)), replace=False)
    for i in spot:
        if flagged[i]:
            expected = p_severity_flagged[drought_pos_map[i]] + 1
        else:
            expected = 0
        assert combined[i] == expected, f"spot check failed at index {i}"

    return combined, flagged


def train_seed(seed):
    print(f"\n{'#' * 70}\nSEED {seed}\n{'#' * 70}")

    m_bin = train_classifier(X_train, y_bin_train, S_train,
                             X_val, y_bin_val, S_val,
                             2, f"stage1 binary seed={seed}", seed, labels=BINARY_LABELS)
    p_bin = predict(m_bin, X_test, S_test)
    metrics_bin = binary_detection_metrics(y_bin_test, p_bin)
    print(f"  binary: Acc {metrics_bin['Accuracy']:.4f} | "
          f"F1 {metrics_bin['F1']:.4f} | Macro F1 {metrics_bin['Macro F1']:.4f}")

    m_sev = train_classifier(X_sev_train, y_sev_train, S_sev_train,
                             X_sev_val, y_sev_val, S_sev_val,
                             3, f"stage2 severity seed={seed}", seed, labels=SEVERITY_LABELS)
    flagged = p_bin == 1
    p_sev_flagged = predict(m_sev, X_test[flagged], S_test[flagged])
    assert len(p_sev_flagged) == int(flagged.sum())
    metrics_sev_train_side = multiclass_metrics(y_sev_test, predict(m_sev, X_sev_test, S_sev_test),
                                                labels=SEVERITY_LABELS)
    print(f"  severity (on true droughts): Acc {metrics_sev_train_side['Accuracy']:.4f} | "
          f"Macro F1 {metrics_sev_train_side['Macro F1']:.4f}")

    m_single = train_classifier(X_train, y_train, S_train,
                                X_val, y_val, S_val,
                                4, f"single 4-class seed={seed}", seed)
    p_single = predict(m_single, X_test, S_test)
    metrics_single = multiclass_metrics(y_test, p_single, labels=FOUR_CLASS_LABELS)
    print(f"  single: Acc {metrics_single['Accuracy']:.4f} | "
          f"Macro F1 {metrics_single['Macro F1']:.4f}")

    combined, flagged = assemble_combined(y_test, p_bin, p_sev_flagged)
    metrics_combined = multiclass_metrics(y_test, combined, labels=FOUR_CLASS_LABELS)
    crit = task2_critical_errors(y_test, combined)
    print(f"  combined: Acc {metrics_combined['Accuracy']:.4f} | "
          f"Macro F1 {metrics_combined['Macro F1']:.4f} | "
          f"flagged={int(flagged.sum())}/{len(y_test)} (true droughts={int(y_bin_test.sum())})")

    return {
        "seed": seed,
        "single": metrics_single,
        "binary": metrics_bin,
        "severity": metrics_sev_train_side,
        "combined": metrics_combined,
        "critical_errors": crit,
    }


def save_results(results):
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {RESULT_PATH}")


def main():
    global X_train, y_train, S_train, X_val, y_val, S_val, X_test, y_test, S_test
    global y_bin_train, y_bin_val, y_bin_test
    global X_sev_train, y_sev_train, S_sev_train
    global X_sev_val, y_sev_val, S_sev_val
    global X_sev_test, y_sev_test, S_sev_test

    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} not found -- run load_data_task1_kz_features.py first.")
    data = np.load(DATA_PATH, allow_pickle=True)
    X_train, y_train, S_train = data["X_train"], data["y_train"], data["S_train"]
    X_val, y_val, S_val = data["X_val"], data["y_val"], data["S_val"]
    X_test, y_test, S_test = data["X_test"], data["y_test"], data["S_test"]

    y_bin_train = (y_train > 0).astype(np.int64)
    y_bin_val = (y_val > 0).astype(np.int64)
    y_bin_test = (y_test > 0).astype(np.int64)

    sev_mask_tr = y_bin_train == 1
    sev_mask_va = y_bin_val == 1
    sev_mask_te = y_bin_test == 1
    X_sev_train, y_sev_train, S_sev_train = (
        X_train[sev_mask_tr], y_train[sev_mask_tr] - 1, S_train[sev_mask_tr])
    X_sev_val, y_sev_val, S_sev_val = (
        X_val[sev_mask_va], y_val[sev_mask_va] - 1, S_val[sev_mask_va])
    X_sev_test, y_sev_test, S_sev_test = (
        X_test[sev_mask_te], y_test[sev_mask_te] - 1, S_test[sev_mask_te])

    print(f"Device: {DEVICE}")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Binary train dist: {np.bincount(y_bin_train, minlength=2)}")
    print(f"Severity train dist: {np.bincount(y_sev_train, minlength=3)}")

    results = {"runs": [], "single_stage": [], "two_stage_binary": [],
               "two_stage_severity": [], "two_stage_combined": [],
               "critical_errors": []}
    for seed in SEEDS:
        entry = train_seed(seed)
        results["runs"].append(entry)
        key_map = {
            "single_stage": "single",
            "two_stage_binary": "binary",
            "two_stage_severity": "severity",
            "two_stage_combined": "combined",
        }
        for key, entry_key in key_map.items():
            results[key].append({"seed": seed, **entry[entry_key]})
        results["critical_errors"].append({"seed": seed, **entry["critical_errors"]})
        save_results(results)

    results["summary"] = {
        "single_stage": summarize_runs(results["single_stage"], SUMMARY_METRICS),
        "two_stage_binary": summarize_runs(
            results["two_stage_binary"],
            ["Accuracy", "Precision", "Recall", "F1", "Macro F1"]),
        "two_stage_severity": summarize_runs(results["two_stage_severity"], SUMMARY_METRICS),
        "two_stage_combined": summarize_runs(results["two_stage_combined"], SUMMARY_METRICS),
        "critical_errors": summarize_runs(results["critical_errors"], CRITICAL_ERROR_KEYS),
    }
    save_results(results)

    print("\n" + "=" * 70)
    print("TASK 2 SUMMARY (mean over seeds)")
    print("=" * 70)
    for name, key in [("Baseline single 4-class", "single_stage"),
                      ("Two-stage combined", "two_stage_combined")]:
        m = results["summary"][key].get("Macro F1", {})
        a = results["summary"][key].get("Accuracy", {})
        print(f"  {name:<24}: Macro F1 {m.get('mean', float('nan')):.4f} "
              f"+/- {m.get('std', 0.0):.4f} | Accuracy {a.get('mean', float('nan')):.4f}")


if __name__ == "__main__":
    main()