"""Task 1 / Scenario D dataset builder.

Reuses the Scenario C loader (same NetCDF, same Kazakhstan bbox, same seed=42
500-cell sampling) and then engineers 22 SPEI-derived features per timestep
into 6-month windows, exactly per the Task 1 spec's build_features().

Input : X = (N, 6, 22) engineered features, window months [t-5 .. t]
Target: y = SPEI class (0-3) of the NEXT month (t+1)
Policy: skip any window where ANY of the 6x22 features is NaN, or the
        target month value is NaN; require >= 18 months of history before the
        first usable window (12 for rolling-12 reach + 6 for the window).
Split : per-location chronological 70/15/15.
Saves sequences_task1_kz_features.npz with the standardized key set.
"""

import numpy as np

from load_data_task1_kz_raw import load_and_sample_spei, spei_to_class_4
from model_lstm_attention import make_zero_static

OUT_PATH = "sequences_task1_kz_features.npz"

SEQ_LEN_MONTHS = 6
FEATURE_REACH = 12            # 12-month lookback for rolling-12 / change / counts
MIN_HISTORY = FEATURE_REACH + SEQ_LEN_MONTHS   # 18 months before first window
N_FEATURES = 22               # see FEATURE_NAMES below

CLASS_LABELS_KZ = ["None", "Mild", "Moderate", "Severe+"]

FEATURE_NAMES = [
    "spei_raw",
    "lag_1m", "lag_2m", "lag_3m", "lag_6m",
    "rolling_mean_3m", "rolling_mean_6m", "rolling_mean_12m",
    "rolling_min_3m", "rolling_min_6m", "rolling_min_12m",
    "rolling_max_3m", "rolling_max_6m", "rolling_max_12m",
    "rolling_std_3m", "rolling_std_6m", "rolling_std_12m",
    "change_1m", "change_3m", "change_6m",
    "neg_spei_count_6m", "neg_spei_count_12m",
]


def build_features(full_series):
    """Per-month 22-feature engineering, written exactly per the Task 1 spec."""
    n = len(full_series)
    feats_all = np.full((n, N_FEATURES), np.nan, dtype=np.float64)

    for t in range(n):
        feats = np.full(N_FEATURES, np.nan)
        val_t = full_series[t]

        feats[0] = val_t

        feats[1] = full_series[t - 1] if t >= 1 else np.nan
        feats[2] = full_series[t - 2] if t >= 2 else np.nan
        feats[3] = full_series[t - 3] if t >= 3 else np.nan
        feats[4] = full_series[t - 6] if t >= 6 else np.nan

        # 3-month / 6-month / 12-month trailing windows ending at t
        for span, i_mean, i_min, i_max, i_std in [
            (3, 5, 8, 11, 14),
            (6, 6, 9, 12, 15),
            (12, 7, 10, 13, 16),
        ]:
            window = full_series[max(0, t - span + 1):t + 1]
            if len(window) > 0:
                feats[i_mean] = float(np.nanmean(window))
                feats[i_min] = float(np.nanmin(window))
                feats[i_max] = float(np.nanmax(window))
                feats[i_std] = float(np.nanstd(window))

        feats[17] = (val_t - full_series[t - 1]) if t >= 1 else np.nan   # change_1m
        feats[18] = (val_t - full_series[t - 3]) if t >= 3 else np.nan   # change_3m
        feats[19] = (val_t - full_series[t - 6]) if t >= 6 else np.nan   # change_6m

        feats[20] = float(np.sum(full_series[max(0, t - 5):t + 1] < 0))
        feats[21] = float(np.sum(full_series[max(0, t - 11):t + 1] < 0))

        feats_all[t] = feats

    return feats_all


def build_per_location_sequences(values):
    all_X = []
    all_y = []
    loc_seq_counts = []
    for loc_i in range(values.shape[1]):
        series = values[:, loc_i]
        if len(series) < MIN_HISTORY:
            loc_seq_counts.append(0)
            continue

        feats = build_features(series)
        loc_X = []
        loc_y = []
        for t in range(FEATURE_REACH + SEQ_LEN_MONTHS - 1, len(series) - 1):
            window = feats[t - (SEQ_LEN_MONTHS - 1):t + 1]
            target_val = series[t + 1]
            if np.isnan(window).any() or np.isnan(target_val):
                continue
            loc_X.append(window)
            loc_y.append(spei_to_class_4(target_val))

        loc_seq_counts.append(len(loc_X))
        all_X.append(loc_X)
        all_y.append(loc_y)

    train_X, train_y = [], []
    val_X, val_y = [], []
    test_X, test_y = [], []
    for loc_X, loc_y in zip(all_X, all_y):
        n_loc = len(loc_X)
        if n_loc == 0:
            continue
        train_end = int(n_loc * 0.7)
        val_end = int(n_loc * 0.85)
        train_X.extend(loc_X[:train_end])
        train_y.extend(loc_y[:train_end])
        val_X.extend(loc_X[train_end:val_end])
        val_y.extend(loc_y[train_end:val_end])
        test_X.extend(loc_X[val_end:])
        test_y.extend(loc_y[val_end:])

    X_train = np.array(train_X, dtype=np.float32).reshape(-1, SEQ_LEN_MONTHS, N_FEATURES)
    y_train = np.array(train_y, dtype=np.int64)
    X_val = np.array(val_X, dtype=np.float32).reshape(-1, SEQ_LEN_MONTHS, N_FEATURES)
    y_val = np.array(val_y, dtype=np.int64)
    X_test = np.array(test_X, dtype=np.float32).reshape(-1, SEQ_LEN_MONTHS, N_FEATURES)
    y_test = np.array(test_y, dtype=np.int64)

    print(f"\nLocations used: {np.sum(np.array(loc_seq_counts) > 0)} / {len(loc_seq_counts)}")
    print(f"Sequences per location: min={np.min(loc_seq_counts)}, "
          f"median={np.median(loc_seq_counts):.0f}, max={np.max(loc_seq_counts)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_and_guard(X_train, X_val, X_test):
    flat_tr = X_train.reshape(-1, N_FEATURES)
    mean = np.nanmean(flat_tr, axis=0).astype(np.float32)
    std = np.nanstd(flat_tr, axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if np.isnan(arr).any():
            raise SystemExit(f"[ERROR] {name} still contains NaN after normalization -- aborting save.")
    return X_train, X_val, X_test, mean, std


def main():
    print("Loading spei_6.nc (reusing Scenario C loader: same bbox, seed=42, 500 cells) ...")
    values = load_and_sample_spei()

    X_train, y_train, X_val, y_val, X_test, y_test = build_per_location_sequences(values)
    print(f"\nBuilt sequences: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")

    if len(X_train) < 100 or len(X_test) == 0:
        print("\n[WARNING] Very few sequences produced -- double check the "
              "bounding box, feature engineering, or NaN-skip policy above.")

    X_train, X_val, X_test, mean, std = normalize_and_guard(X_train, X_val, X_test)

    S_train = make_zero_static(len(X_train))
    S_val = make_zero_static(len(X_val))
    S_test = make_zero_static(len(X_test))

    for name, y in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        counts = np.bincount(y, minlength=4)
        print(f"{name} class distribution: {counts} ({CLASS_LABELS_KZ})")

    np.savez(
        OUT_PATH,
        X_train=X_train, y_train=y_train, S_train=S_train,
        X_val=X_val,     y_val=y_val,     S_val=S_val,
        X_test=X_test,   y_test=y_test,   S_test=S_test,
        means=mean, stds=std,
        feature_names=np.array(FEATURE_NAMES),
    )

    print(f"\nSaved {OUT_PATH}")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  X shape: {X_train.shape} ({SEQ_LEN_MONTHS} months, {N_FEATURES} features)")
    print(f"  Feature names: {FEATURE_NAMES}")
    print(f"  means/stds: {mean.tolist()}")
    print(f"  S shape: {S_train.shape} (zero-filled soil placeholder)")
    print(f"[OK] No NaN values remain in X_train/X_val/X_test.")


if __name__ == "__main__":
    main()