import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = "results.json"
DATA_PATH = "sequences_kazakhstan_features.npz"
MODEL_PATH = "model_kazakhstan_features.pt"
N_CLASSES = 4
CLASS_LABELS = ["None", "Mild", "Moderate", "Severe+"]

PROJ_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ = 24, 64, 1, 0.4, 8
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_results(model_name, metrics, extra=None):
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    entry = {k: float(v) if isinstance(v, (float, np.floating)) else v
             for k, v in metrics.items()}
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


def compute_baseline_accuracy(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, sb, yb in loader:
            xb, sb = xb.to(DEVICE), sb.to(DEVICE)
            logits = model(xb, sb)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    return accuracy_score(trues, preds), f1_score(trues, preds, average="macro", zero_division=0)


def permutation_importance_features(model, X, S, y, feature_names, n_repeats=5):
    """Permute each feature across all timesteps and measure F1 drop."""
    model.eval()
    loader = DataLoader(DroughtDataset(X, S, y), BATCH_SIZE * 2)
    base_acc, base_f1 = compute_baseline_accuracy(model, loader)

    n_features = X.shape[-1]
    importance = np.zeros((n_features, n_repeats))

    for feat_idx in range(n_features):
        for rep in range(n_repeats):
            X_perm = X.copy()
            perm_order = np.random.permutation(len(X))
            X_perm[:, :, feat_idx] = X[perm_order, :, feat_idx]

            perm_loader = DataLoader(DroughtDataset(X_perm, S, y), BATCH_SIZE * 2)
            _, perm_f1 = compute_baseline_accuracy(model, perm_loader)
            importance[feat_idx, rep] = base_f1 - perm_f1

    mean_imp = importance.mean(axis=1)
    std_imp = importance.std(axis=1)

    print(f"\n  Baseline: Accuracy={base_acc:.4f}, Macro F1={base_f1:.4f}")
    print(f"\n  Feature permutation importance (F1 drop = importance):")
    order = np.argsort(mean_imp)[::-1]
    for i in order:
        print(f"    {feature_names[i]:<25} {mean_imp[i]:>+.4f} +/- {std_imp[i]:.4f}")

    return mean_imp, std_imp, base_f1


def permutation_importance_timesteps(model, X, S, y, n_timesteps=18, n_repeats=5):
    """Permute each timestep (month) and measure F1 drop."""
    model.eval()
    loader = DataLoader(DroughtDataset(X, S, y), BATCH_SIZE * 2)
    base_acc, base_f1 = compute_baseline_accuracy(model, loader)

    importance = np.zeros((n_timesteps, n_repeats))

    for t in range(n_timesteps):
        for rep in range(n_repeats):
            X_perm = X.copy()
            perm_order = np.random.permutation(len(X))
            X_perm[:, t, :] = X[perm_order, t, :]

            perm_loader = DataLoader(DroughtDataset(X_perm, S, y), BATCH_SIZE * 2)
            _, perm_f1 = compute_baseline_accuracy(model, perm_loader)
            importance[t, rep] = base_f1 - perm_f1

    mean_imp = importance.mean(axis=1)
    std_imp = importance.std(axis=1)

    print(f"\n  Timestep permutation importance:")
    month_labels = [f"Month {i+1}" for i in range(n_timesteps)]
    order = np.argsort(mean_imp)[::-1]
    for i in order:
        print(f"    {month_labels[i]:<12} {mean_imp[i]:>+.4f} +/- {std_imp[i]:.4f}")

    return mean_imp, std_imp


def plot_feature_importance(mean_imp, std_imp, feature_names, save_path="shap_kz_feature_importance.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = np.argsort(mean_imp)
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in mean_imp[order]]
    ax.barh(range(len(order)), mean_imp[order], xerr=std_imp[order],
            color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.set_xlabel("Macro F1 drop (importance)")
    ax.set_title("Kazakhstan Model: Feature Permutation Importance")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved {save_path}")


def plot_timestep_importance(mean_imp, std_imp, save_path="shap_kz_timestep_importance.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(mean_imp))
    ax.bar(x, mean_imp, yerr=std_imp, color="#3498db", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"M{i+1}" for i in range(len(mean_imp))], rotation=45)
    ax.set_ylabel("Macro F1 drop (importance)")
    ax.set_xlabel("Month in input window (M1 = oldest, M18 = most recent)")
    ax.set_title("Kazakhstan Model: Timestep Permutation Importance")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_per_class_importance(model, X, S, y, feature_names, save_path="shap_kz_per_class.png"):
    model.eval()
    n_features = X.shape[-1]
    class_f1_drop = np.zeros((N_CLASSES, n_features))

    for cls_idx in range(N_CLASSES):
        cls_mask = y == cls_idx
        if cls_mask.sum() < 50:
            continue
        X_cls, S_cls, y_cls = X[cls_mask], S[cls_mask], y[cls_mask]

        loader = DataLoader(DroughtDataset(X_cls, S_cls, y_cls), BATCH_SIZE * 2)
        preds_orig = []
        with torch.no_grad():
            for xb, sb, yb in loader:
                xb, sb = xb.to(DEVICE), sb.to(DEVICE)
                preds_orig.extend(model(xb, sb).argmax(1).cpu().numpy())
        base_f1 = f1_score(y_cls, preds_orig, average="macro", zero_division=0)

        for feat_idx in range(n_features):
            X_perm = X_cls.copy()
            perm_order = np.random.permutation(len(X_cls))
            X_perm[:, :, feat_idx] = X_cls[perm_order, :, feat_idx]

            perm_loader = DataLoader(DroughtDataset(X_perm, S_cls, y_cls), BATCH_SIZE * 2)
            preds_perm = []
            with torch.no_grad():
                for xb, sb, yb in perm_loader:
                    xb, sb = xb.to(DEVICE), sb.to(DEVICE)
                    preds_perm.extend(model(xb, sb).argmax(1).cpu().numpy())
            perm_f1 = f1_score(y_cls, preds_perm, average="macro", zero_division=0)
            class_f1_drop[cls_idx, feat_idx] = base_f1 - perm_f1

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(class_f1_drop, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_title("Per-Class Feature Importance (F1 drop on permutation)")
    plt.colorbar(im, ax=ax, label="F1 drop")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")

    return class_f1_drop


def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} not found.")
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"{MODEL_PATH} not found -- run training first.")

    data = np.load(DATA_PATH, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    S_test = data["S_test"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    S_val = data["S_val"]
    feature_names = list(data["feature_names"])

    print(f"Device: {DEVICE}")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Test set: {X_test.shape}")

    model = LSTMAttention(X_test.shape[-1], S_test.shape[-1], PROJ_SIZE,
                          HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ,
                          N_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Loaded {MODEL_PATH}")

    print(f"\n{'='*60}")
    print(f"FEATURE-LEVEL PERMUTATION IMPORTANCE")
    print(f"{'='*60}")
    feat_imp, feat_std, base_f1 = permutation_importance_features(
        model, X_test, S_test, y_test, feature_names, n_repeats=3)

    plot_feature_importance(feat_imp, feat_std, feature_names)

    print(f"\n{'='*60}")
    print(f"TIMESTEP-LEVEL PERMUTATION IMPORTANCE")
    print(f"{'='*60}")
    ts_imp, ts_std = permutation_importance_timesteps(
        model, X_test, S_test, y_test, n_timesteps=X_test.shape[1], n_repeats=3)

    plot_timestep_importance(ts_imp, ts_std)

    print(f"\n{'='*60}")
    print(f"PER-CLASS FEATURE IMPORTANCE")
    print(f"{'='*60}")
    per_class = plot_per_class_importance(model, X_test, S_test, y_test, feature_names)

    save_results("KZ explainability (permutation importance)",
                 {"baseline_f1": base_f1},
                 extra={"feature_names": feature_names,
                        "feature_importance": feat_imp.tolist(),
                        "timestep_importance": ts_imp.tolist(),
                        "n_repeats": 3})

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    top3 = np.argsort(feat_imp)[::-1][:3]
    print(f"  Top 3 most important features:")
    for i in top3:
        print(f"    {feature_names[i]:<25} (F1 drop: {feat_imp[i]:+.4f})")
    top3_ts = np.argsort(ts_imp)[::-1][:3]
    print(f"  Top 3 most important timesteps:")
    for i in top3_ts:
        print(f"    Month {i+1:<20} (F1 drop: {ts_imp[i]:+.4f})")


if __name__ == "__main__":
    main()
