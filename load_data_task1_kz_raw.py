"""Task 1 / Scenario C dataset builder.

Loads raw 6-month SPEI windows over Kazakhstan's bounding box from the same
NetCDF used by load_data_kazakhstan.py, but:
  * window = months [t-5 .. t],  target = class of month t+1  (spec semantics)
  * per-location chronological 70/15/15 split (spec) instead of one global cut
Saves sequences_task1_kz_raw.npz with the standardized key set.
"""

import numpy as np

from model_lstm_attention import make_zero_static

try:
    import xarray as xr
except ImportError:
    raise SystemExit(
        "This script needs xarray + netCDF4.\n"
        "Install with:  pip install xarray netCDF4"
    )

NC_PATH = "spei_6.nc"
OUT_PATH = "sequences_task1_kz_raw.npz"

KZ_LON_MIN, KZ_LON_MAX = 46.5, 87.3
KZ_LAT_MIN, KZ_LAT_MAX = 40.6, 55.4

SEQ_LEN_MONTHS = 6
MAX_GRID_CELLS = 500
SEED = 42
SPEI_PLAUSIBLE_RANGE = 5.0

FEATURE_NAMES = ["spei_raw"]
CLASS_LABELS_KZ = ["None", "Mild", "Moderate", "Severe+"]


def spei_to_class_4(spei_value):
    if spei_value >= -1.0:
        return 0
    elif spei_value >= -1.5:
        return 1
    elif spei_value >= -2.0:
        return 2
    else:
        return 3


def load_and_sample_spei(path=NC_PATH):
    ds = xr.open_dataset(path)

    spei_var = None
    for candidate in ["spei", "SPEI", "Standardized_Precipitation_Evapotranspiration_Index"]:
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
        print("[Note] Latitude appears descending -- re-slicing in reverse order.")
        kz = ds[spei_var].sel(
            {lon_name: slice(KZ_LON_MIN, KZ_LON_MAX),
             lat_name: slice(KZ_LAT_MAX, KZ_LAT_MIN)}
        )

    print(f"Cropped shape: {kz.shape}  (dims: {kz.dims})")

    raw_min = float(np.nanmin(kz.values))
    raw_max = float(np.nanmax(kz.values))
    print(f"Raw SPEI value range in cropped region: [{raw_min:.3f}, {raw_max:.3f}]")

    n_time = kz.sizes.get("time", 0)
    if n_time < SEQ_LEN_MONTHS + 1:
        raise SystemExit(
            f"Only {n_time} time steps available after cropping -- need at "
            f"least {SEQ_LEN_MONTHS + 1}. Check the bounding box / file."
        )

    stacked = kz.stack(location=(lat_name, lon_name))
    values = stacked.values
    print(f"Total grid cells in Kazakhstan bbox: {values.shape[1]}")

    n_before = np.isnan(values).sum()
    values = np.where(np.abs(values) > SPEI_PLAUSIBLE_RANGE, np.nan, values)
    n_after = np.isnan(values).sum()
    if n_after > n_before:
        print(f"[Note] Sanitized {n_after - n_before} out-of-range sentinel "
              f"values (|SPEI| > {SPEI_PLAUSIBLE_RANGE}) into proper NaN.")

    valid_mask = ~np.isnan(values).all(axis=0)
    values = values[:, valid_mask]
    n_valid = values.shape[1]
    print(f"Valid (non-NaN) grid cells: {n_valid}")

    if n_valid == 0:
        raise SystemExit("No valid grid cells found -- check variable/coord names or bbox.")

    rng = np.random.default_rng(SEED)
    n_sample = min(MAX_GRID_CELLS, n_valid)
    sample_idx = rng.choice(n_valid, size=n_sample, replace=False)
    values = values[:, sample_idx]
    print(f"Sampled {n_sample} grid cells (seed={SEED})")

    return values


def build_per_location_sequences(values):
    all_X = []
    all_y = []
    loc_seq_counts = []
    for loc_i in range(values.shape[1]):
        series = values[:, loc_i]
        loc_X = []
        loc_y = []
        for t in range(SEQ_LEN_MONTHS - 1, len(series) - 1):
            window = series[t - (SEQ_LEN_MONTHS - 1):t + 1]
            target = series[t + 1]
            if np.isnan(window).any() or np.isnan(target):
                continue
            loc_X.append(window)
            loc_y.append(spei_to_class_4(target))
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

    X_train = np.array(train_X, dtype=np.float32).reshape(-1, SEQ_LEN_MONTHS, 1)
    y_train = np.array(train_y, dtype=np.int64)
    X_val = np.array(val_X, dtype=np.float32).reshape(-1, SEQ_LEN_MONTHS, 1)
    y_val = np.array(val_y, dtype=np.int64)
    X_test = np.array(test_X, dtype=np.float32).reshape(-1, SEQ_LEN_MONTHS, 1)
    y_test = np.array(test_y, dtype=np.int64)

    print(f"\nLocations used: {np.sum(np.array(loc_seq_counts) > 0)} / {len(loc_seq_counts)}")
    print(f"Sequences per location: min={np.min(loc_seq_counts)}, "
          f"median={np.median(loc_seq_counts):.0f}, max={np.max(loc_seq_counts)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_and_guard(X_train, X_val, X_test):
    mean = np.nanmean(X_train.reshape(-1, len(FEATURE_NAMES)), axis=0).astype(np.float32)
    std = np.nanstd(X_train.reshape(-1, len(FEATURE_NAMES)), axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if np.isnan(arr).any():
            raise SystemExit(f"[ERROR] {name} still contains NaN after normalization -- aborting save.")
    return X_train, X_val, X_test, mean, std


def main():
    print(f"Loading {NC_PATH} ...")
    print("(This is a large file -- may take a minute to open.)")
    values = load_and_sample_spei()

    X_train, y_train, X_val, y_val, X_test, y_test = build_per_location_sequences(values)
    print(f"\nBuilt sequences: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")

    if len(X_train) < 100 or len(X_test) == 0:
        print("\n[WARNING] Very few sequences produced -- double check the "
              "bounding box and variable names above before proceeding.")

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
    print(f"  X shape: {X_train.shape} (6 months, 1 raw feature '{FEATURE_NAMES[0]}')")
    print(f"  S shape: {S_train.shape} (zero-filled soil placeholder)")
    print(f"[OK] No NaN values remain in X_train/X_val/X_test.")


if __name__ == "__main__":
    main()