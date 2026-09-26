import numpy as np

try:
    import xarray as xr
except ImportError:
    raise SystemExit(
        "This script needs xarray + netCDF4.\n"
        "Install with:  pip install xarray netCDF4"
    )

NC_PATH = "spei_6.nc"
OUT_PATH = "sequences_kazakhstan_features.npz"

KZ_LON_MIN, KZ_LON_MAX = 46.5, 87.3
KZ_LAT_MIN, KZ_LAT_MAX = 40.6, 55.4

# Need enough history for a 12-month rolling stat PLUS the 6-month
# prediction window itself, so we use an 18-month total lookback.
SEQ_LEN_MONTHS = 18
MAX_GRID_CELLS = 500
SEED = 42

ROLLING_WINDOWS = [3, 6, 12]   # months
LAG_STEPS = [1, 3, 6]          # months

FEATURE_NAMES = (
    ["spei_raw"]
    + [f"rolling_min_{w}m" for w in ROLLING_WINDOWS]
    + [f"rolling_max_{w}m" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}m" for w in ROLLING_WINDOWS]
    + [f"lag_{l}m" for l in LAG_STEPS]
    + ["rate_of_change"]
)
N_FEATURES = len(FEATURE_NAMES)  # 1 + 3 + 3 + 3 + 3 + 1 = 14

def spei_to_class_4(spei_value):
    if spei_value >= -1.0:
        return 0   # None
    elif spei_value >= -1.5:
        return 1   # Mild
    elif spei_value >= -2.0:
        return 2   # Moderate
    else:
        return 3   # Severe+

CLASS_LABELS_KZ = ["None", "Mild", "Moderate", "Severe+"]

def build_engineered_features(full_series, t):
    """For a single location's SPEI series, build the 14 engineered
    features that describe the state AT month t (using only data up to
    and including month t -- no lookahead)."""
    feats = np.full(N_FEATURES, np.nan, dtype=np.float32)
    feats[0] = full_series[t]  # raw SPEI

    idx = 1
    for w in ROLLING_WINDOWS:
        window = full_series[max(0, t - w + 1):t + 1]
        feats[idx] = np.nanmin(window) if len(window) > 0 else np.nan
        idx += 1
    for w in ROLLING_WINDOWS:
        window = full_series[max(0, t - w + 1):t + 1]
        feats[idx] = np.nanmax(window) if len(window) > 0 else np.nan
        idx += 1
    for w in ROLLING_WINDOWS:
        window = full_series[max(0, t - w + 1):t + 1]
        feats[idx] = np.nanstd(window) if len(window) > 0 else np.nan
        idx += 1
    for l in LAG_STEPS:
        feats[idx] = full_series[t - l] if t - l >= 0 else np.nan
        idx += 1
    feats[idx] = full_series[t] - full_series[t - 1] if t >= 1 else np.nan

    return feats

def main():
    print(f"Loading {NC_PATH} ...")
    ds = xr.open_dataset(NC_PATH)

    spei_var = None
    for candidate in ["spei", "SPEI"]:
        if candidate in ds.data_vars:
            spei_var = candidate
            break
    if spei_var is None:
        spei_var = list(ds.data_vars)[0]

    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"

    kz = ds[spei_var].sel(
        {lon_name: slice(KZ_LON_MIN, KZ_LON_MAX),
         lat_name: slice(KZ_LAT_MIN, KZ_LAT_MAX)}
    )
    if kz.sizes[lat_name] == 0:
        kz = ds[spei_var].sel(
            {lon_name: slice(KZ_LON_MIN, KZ_LON_MAX),
             lat_name: slice(KZ_LAT_MAX, KZ_LAT_MIN)}
        )

    print(f"Cropped shape: {kz.shape} (dims: {kz.dims})")
    raw_min = float(np.nanmin(kz.values))
    raw_max = float(np.nanmax(kz.values))
    print(f"Raw SPEI value range: [{raw_min:.3f}, {raw_max:.3f}]")

    stacked = kz.stack(location=(lat_name, lon_name))
    values = stacked.values  # (time, n_locations)

    # Sanitize sentinel values (same fix as load_data_kazakhstan.py)
    SPEI_PLAUSIBLE_RANGE = 5.0
    values = np.where(np.abs(values) > SPEI_PLAUSIBLE_RANGE, np.nan, values)

    valid_mask = ~np.isnan(values).all(axis=0)
    values = values[:, valid_mask]
    n_valid = values.shape[1]
    print(f"Valid grid cells: {n_valid}")

    rng = np.random.default_rng(SEED)
    n_sample = min(MAX_GRID_CELLS, n_valid)
    sample_idx = rng.choice(n_valid, size=n_sample, replace=False)
    values = values[:, sample_idx]
    print(f"Sampled {n_sample} grid cells (seed={SEED})")

    print(f"\nBuilding {N_FEATURES} engineered features per timestep: {FEATURE_NAMES}")

    X, y = [], []
    for loc_i in range(values.shape[1]):
        series = values[:, loc_i]
        # Need SEQ_LEN_MONTHS of history before t, and t itself must have
        # enough prior context (>=12 months) for the rolling-12m stat to
        # be meaningful, so start at max(SEQ_LEN_MONTHS, 12).
        start_t = max(SEQ_LEN_MONTHS, 12)
        for t in range(start_t, len(series) - 1):
            # Build the engineered feature sequence for the past
            # SEQ_LEN_MONTHS months (t-SEQ_LEN_MONTHS+1 .. t)
            window_feats = []
            skip = False
            for tau in range(t - SEQ_LEN_MONTHS + 1, t + 1):
                f = build_engineered_features(series, tau)
                if np.isnan(f).any():
                    skip = True
                    break
                window_feats.append(f)
            if skip:
                continue

            target = series[t + 1]  # predict NEXT month's SPEI class
            if np.isnan(target):
                continue

            X.append(window_feats)
            y.append(spei_to_class_4(target))

    X = np.array(X, dtype=np.float32)   # (N, SEQ_LEN_MONTHS, N_FEATURES)
    y = np.array(y, dtype=np.int64)
    print(f"\nBuilt {len(X)} sequences of shape (seq_len={SEQ_LEN_MONTHS}, features={N_FEATURES})")
    print(f"Class distribution: {np.bincount(y, minlength=4)} ({CLASS_LABELS_KZ})")

    if len(X) < 100:
        print("\n[WARNING] Very few sequences produced. The 18-month lookback "
              "requirement is stricter than the original 6-month pipeline, "
              "so expect meaningfully fewer usable sequences.")

    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    # Normalize each feature independently (fit on train only)
    mean = X_train.reshape(-1, N_FEATURES).mean(axis=0).astype(np.float32)
    std  = X_train.reshape(-1, N_FEATURES).std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_test  = (X_test - mean) / std

    assert not np.isnan(X_train).any(), "X_train has NaNs -- aborting."
    assert not np.isnan(X_val).any(),   "X_val has NaNs -- aborting."
    assert not np.isnan(X_test).any(),  "X_test has NaNs -- aborting."
    print("\n[OK] Verified no NaN values remain.")

    soil_dim = 31
    S_train = np.zeros((len(X_train), soil_dim), dtype=np.float32)
    S_val   = np.zeros((len(X_val), soil_dim), dtype=np.float32)
    S_test  = np.zeros((len(X_test), soil_dim), dtype=np.float32)

    np.savez(OUT_PATH,
             X_train=X_train, y_train=y_train, S_train=S_train,
             X_val=X_val,     y_val=y_val,     S_val=S_val,
             X_test=X_test,   y_test=y_test,   S_test=S_test,
             feature_names=np.array(FEATURE_NAMES),
             class_labels=np.array(CLASS_LABELS_KZ),
             means=mean, stds=std,
             seq_len_months=SEQ_LEN_MONTHS)

    print(f"\nSaved {OUT_PATH}")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Shape: {X_train.shape}  ({N_FEATURES} engineered features vs. "
          f"1 raw feature in the original Kazakhstan pipeline)")

if __name__ == "__main__":
    main()
