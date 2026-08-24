import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE      = torch.device("cpu")   # CPU required -- cuDNN LSTM can't backprop in eval
CLASS_SCHEME= 4        # <-- must match the scheme used to train model.pt
N_CLASSES   = CLASS_SCHEME
DATA_PATH   = f"sequences_{CLASS_SCHEME}.npz"
PROJ_SIZE   = 32
HIDDEN_SIZE = 96
N_LAYERS    = 1
DROPOUT     = 0.4
SOIL_PROJ   = 16
N_BG        = 100   # background samples
N_EXP       = 50    # samples to explain

FEATURE_NAMES = [
    "PRECTOT","PS","QV2M","T2M","T2MDEW","T2MWET",
    "T2M_MAX","T2M_MIN","T2M_RANGE","TS",
    "WS10M","WS10M_MAX","WS10M_MIN","WS10M_RANGE",
    "WS50M","WS50M_MAX","WS50M_MIN","WS50M_RANGE",
]
FEATURE_LABELS = [
    "Precipitation","Surface Pressure","Humidity (2m)","Temperature (2m)",
    "Dew Point","Wet Bulb Temp","Max Temp","Min Temp","Temp Range",
    "Skin Temp","Wind Speed (10m)","Max Wind (10m)","Min Wind (10m)",
    "Wind Range (10m)","Wind Speed (50m)","Max Wind (50m)",
    "Min Wind (50m)","Wind Range (50m)",
]

CLASS_LABELS = {
    4: ["None", "Mild", "Moderate", "Severe+"],
    5: ["None", "D0", "D1", "D2", "D3+"],
    6: ["None", "D0", "D1", "D2", "D3", "D4"],
}[CLASS_SCHEME]
MOST_SEVERE_CLASS = N_CLASSES - 1  # last class index = most severe

class LSTMAttention(nn.Module):
    """Must match train_model.py's architecture exactly, or model.pt
    weights won't load correctly."""
    def __init__(self, input_size, soil_size, proj_size, hidden_size,
                 n_layers, dropout, soil_proj, n_classes):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_size), nn.LayerNorm(proj_size),
            nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.lstm   = nn.LSTM(proj_size, hidden_size, n_layers, batch_first=True)
        self.attn_w = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)
        self.drop   = nn.Dropout(dropout)
        self.soil_proj = nn.Sequential(
            nn.Linear(soil_size, soil_proj), nn.LayerNorm(soil_proj),
            nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.fc1  = nn.Linear(hidden_size + soil_proj, 32)
        self.fc2  = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x, s):
        B, T, F = x.shape
        x = self.proj(x.reshape(B*T, F)).reshape(B, T, -1)
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1), dim=1)
        ctx = (out * w.unsqueeze(-1)).sum(dim=1)
        soil_ctx = self.soil_proj(s)
        combined = torch.cat([self.drop(ctx), soil_ctx], dim=1)
        return self.fc2(self.relu(self.fc1(combined)))

class FlatWrapper(nn.Module):
    """SHAP's GradientExplainer needs a single flat input tensor. This
    wrapper takes one concatenated [weather_flat | soil] vector, splits
    it back into the two branches, and forwards through the real model."""
    def __init__(self, model, seq_len, n_weather_feats, n_soil_feats):
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.n_weather_feats = n_weather_feats
        self.n_soil_feats = n_soil_feats
        self.weather_size = seq_len * n_weather_feats

    def forward(self, x):
        weather_flat = x[:, :self.weather_size]
        soil = x[:, self.weather_size:]
        weather = weather_flat.reshape(-1, self.seq_len, self.n_weather_feats)
        return self.model(weather, soil)

def main():
    data    = np.load(DATA_PATH, allow_pickle=True)
    X_test  = data["X_test"].astype(np.float32)
    S_test  = data["S_test"].astype(np.float32)
    y_test  = data["y_test"]
    soil_names = list(data["soil_names"]) if "soil_names" in data else \
                 [f"soil_{i}" for i in range(S_test.shape[1])]
    print(f"Test: {X_test.shape}, soil: {S_test.shape}, "
          f"classes: {np.bincount(y_test, minlength=N_CLASSES)}")

    seq_len = X_test.shape[1]
    n_weather_feats = X_test.shape[2]
    n_soil_feats = S_test.shape[1]

    model = LSTMAttention(n_weather_feats, n_soil_feats, PROJ_SIZE, HIDDEN_SIZE,
                           N_LAYERS, DROPOUT, SOIL_PROJ, N_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
    model.train()  # must be train mode for LSTM gradient flow

    wrapper = FlatWrapper(model, seq_len, n_weather_feats, n_soil_feats).to(DEVICE)
    wrapper.train()

    # Flatten weather + soil into one combined vector per sample so
    # GradientExplainer can treat this as a single-input model.
    X_flat = X_test.reshape(len(X_test), -1)
    combined = np.concatenate([X_flat, S_test], axis=1)
    rng = np.random.default_rng(42)

    bg_idx = rng.choice(len(combined), N_BG, replace=False)
    background = torch.tensor(combined[bg_idx])

    # Explain a balanced sample: half "None" (class 0), half the most
    # severe class available in this scheme.
    none_idx = np.where(y_test == 0)[0]
    sev_idx  = np.where(y_test == MOST_SEVERE_CLASS)[0]
    n_sev_available = len(sev_idx)
    if n_sev_available < N_EXP // 2:
        print(f"  Note: only {n_sev_available} samples of the most severe "
              f"class ({CLASS_LABELS[MOST_SEVERE_CLASS]}) in test set -- "
              f"using all of them.")
    n_sev = min(N_EXP // 2, n_sev_available)
    exp_idx = np.concatenate([
        rng.choice(none_idx, N_EXP - n_sev, replace=False),
        rng.choice(sev_idx, n_sev, replace=False) if n_sev > 0 else np.array([], dtype=int),
    ])
    n_exp_actual = len(exp_idx)
    X_exp = torch.tensor(combined[exp_idx])

    print(f"Running GradientExplainer ({N_BG} background, {n_exp_actual} explain)...")
    explainer   = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(X_exp)
    print("SHAP done.")

    if isinstance(shap_values, list):
        sv = np.stack(shap_values, axis=-1)
    else:
        sv = shap_values

    weather_size = seq_len * n_weather_feats
    sv_weather = sv[:, :weather_size, ...].reshape(n_exp_actual, seq_len, n_weather_feats, N_CLASSES)
    sv_soil    = sv[:, weather_size:, ...].reshape(n_exp_actual, n_soil_feats, N_CLASSES)

    # 1. Global feature importance (weather features only, soil shown separately below)
    imp   = np.abs(sv_weather).mean(axis=(0, 1, 3))
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([FEATURE_LABELS[i] for i in order], imp[order],
            color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(FEATURE_LABELS))))
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"Global Weather Feature Importance\n"
                 f"(LSTM+Attention Drought Classifier, {N_CLASSES}-class)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_summary.png")

    # 1b. Soil feature importance (separate chart, static features)
    soil_imp = np.abs(sv_soil).mean(axis=(0, 2))
    soil_order = np.argsort(soil_imp)
    fig, ax = plt.subplots(figsize=(9, max(4, len(soil_names) * 0.3)))
    ax.barh([soil_names[i] for i in soil_order], soil_imp[soil_order],
            color=plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(soil_names))))
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Soil / Terrain Feature Importance", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("shap_soil_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_soil_importance.png")

    # 2. Case study: most severe class sample
    sev_in_exp = np.where(y_test[exp_idx] == MOST_SEVERE_CLASS)[0]
    if len(sev_in_exp) == 0:
        print("  Note: no severe-class sample available for case study, "
              "using the first explained sample instead.")
        ci = 0
        case_class = int(y_test[exp_idx][0])
    else:
        ci = sev_in_exp[0]
        case_class = MOST_SEVERE_CLASS

    csv = sv_weather[ci, :, :, case_class]
    ts_imp = np.abs(csv).mean(axis=1)
    fi_imp = np.abs(csv).mean(axis=0)
    top5   = np.argsort(fi_imp)[::-1][:5]
    days   = np.arange(1, seq_len + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(days, ts_imp, color="#d62728", lw=1.5)
    axes[0].fill_between(days, ts_imp, alpha=0.2, color="#d62728")
    axes[0].set_xlabel("Day in 90-day window"); axes[0].set_ylabel("Mean |SHAP|")
    axes[0].set_title(f"Case Study: {CLASS_LABELS[case_class]} Drought "
                       f"\u2014 Importance per Timestep", fontweight="bold")
    axes[0].grid(alpha=0.3)
    for j, fi in enumerate(top5):
        axes[1].plot(days, np.abs(csv[:, fi]), label=FEATURE_LABELS[fi],
                     lw=1.5, color=plt.cm.tab10(j / 10))
    axes[1].set_xlabel("Day in 90-day window"); axes[1].set_ylabel("|SHAP Value|")
    axes[1].set_title("Top 5 Features \u2014 SHAP Over Time", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("shap_casestudy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_casestudy.png")

    # 3. Heatmap
    hm = np.abs(sv_weather).mean(axis=(0, 3))
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(hm.T, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(FEATURE_LABELS))); ax.set_yticklabels(FEATURE_LABELS, fontsize=9)
    ax.set_xlabel("Day in 90-day window", fontsize=11)
    ax.set_title("SHAP Importance Heatmap \u2014 Feature \u00d7 Timestep",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig("shap_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_heatmap.png")

    print(f"\n--- TOP 5 WEATHER FEATURES ({N_CLASSES}-class model) ---")
    for rank, i in enumerate(np.argsort(imp)[::-1][:5], 1):
        print(f"  {rank}. {FEATURE_LABELS[i]:<25} SHAP={imp[i]:.6f}")
    print(f"\n--- TOP 5 SOIL FEATURES ---")
    for rank, i in enumerate(np.argsort(soil_imp)[::-1][:5], 1):
        print(f"  {rank}. {soil_names[i]:<25} SHAP={soil_imp[i]:.6f}")

    print("\nDone: shap_summary.png | shap_soil_importance.png | "
          "shap_casestudy.png | shap_heatmap.png")

if __name__ == "__main__":
    main()
