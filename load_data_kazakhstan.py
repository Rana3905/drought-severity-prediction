import numpy as np

try:
    import xarray as xr
except ImportError:
    raise SystemExit(
        "This script needs xarray + netCDF4.\n"
        "Install with:  pip install xarray netCDF4"
    )

NC_PATH = "spei_6.nc"
OUT_PATH = "sequences_kazakhstan.npz"

# Kazakhstan's approximate bounding box (lon_min, lon_max, lat_min, lat_max)
KZ_LON_MIN, KZ_LON_MAX = 46.5, 87.3
KZ_LAT_MIN, KZ_LAT_MAX = 40.6, 55.4

SEQ_LEN_MONTHS = 6          # 6-month input window (matches the SPEI-06 timescale itself)
MAX_GRID_CELLS = 500        # cap, to mirror the "500 counties" sampling & keep runtime/memory sane
SEED = 42

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

def main():
    print(f"Loading {NC_PATH} ...")
    print("(This is a large file -- may take a minute to open.)")
    ds = xr.open_dataset(NC_PATH)

    print("\nDataset variables and dimensions:")
    print(ds)

    # The exact variable/coordinate names can vary slightly between CEDA
    # NetCDF files -- print what's actually inside, then adapt below if needed.
    spei_var = None
    for candidate in ["spei", "SPEI", "Standardized_Precipitation_Evapotranspiration_Index"]:
        if candidate in ds.data_vars:
            spei_var = candidate
            break
    if spei_var is None:
        # fall back to first data variable found
        spei_var = list(ds.data_vars)[0]
        print(f"\n[Note] Guessed SPEI variable name: '{spei_var}' "
              f"(available: {list(ds.data_vars)})")

    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"

    print(f"\nUsing variable='{spei_var}', lat='{lat_name}', lon='{lon_name}'")

    # Crop to Kazakhstan's bounding box
    print(f"\nCropping to Kazakhstan bbox: "
          f"lon [{KZ_LON_MIN}, {KZ_LON_MAX}], lat [{KZ_LAT_MIN}, {KZ_LAT_MAX}]")
    kz = ds[spei_var].sel(
        {lon_name: slice(KZ_LON_MIN, KZ_LON_MAX),
         lat_name: slice(KZ_LAT_MIN, KZ_LAT_MAX)}
    )
    # NOTE: some NetCDF files store latitude descending (90 -> -90) -- if the
    # slice above comes back empty, swap to slice(KZ_LAT_MAX, KZ_LAT_MIN).
    if kz.sizes[lat_name] == 0:
        print("[Note] Latitude appears descending -- re-slicing in reverse order.")
        kz = ds[spei_var].sel(
            {lon_name: slice(KZ_LON_MIN, KZ_LON_MAX),
             lat_name: slice(KZ_LAT_MAX, KZ_LAT_MIN)}
        )

    print(f"Cropped shape: {kz.shape}  (dims: {kz.dims})")

    # Early diagnostic: check the raw value range BEFORE any processing,
    # so a fill-value contamination problem (like the one that broke the
    # first run of this script) is visible immediately, not discovered
    # only after training silently produces NaN losses.
    raw_min = float(np.nanmin(kz.values))
    raw_max = float(np.nanmax(kz.values))
    print(f"Raw SPEI value range in cropped region: [{raw_min:.3f}, {raw_max:.3f}] "
          f"(sanity check: genuine SPEI should be roughly within [-5, 5])")

    n_time = kz.sizes.get("time", 0)
    if n_time < SEQ_LEN_MONTHS + 1:
        raise SystemExit(
            f"Only {n_time} time steps available after cropping -- need at "
            f"least {SEQ_LEN_MONTHS + 1}. Check the bounding box / file."
        )

    # Flatten spatial grid into a list of individual "location" time series,
    # analogous to how load_data.py treats each US county as one series.
    stacked = kz.stack(location=(lat_name, lon_name))   # (time, location)
    values = stacked.values                              # numpy array (time, n_locations)
    n_locations_total = values.shape[1]
    print(f"\nTotal grid cells in Kazakhstan bbox: {n_locations_total}")


    SPEI_PLAUSIBLE_RANGE = 5.0
    n_before = np.isnan(values).sum()
    values = np.where(np.abs(values) > SPEI_PLAUSIBLE_RANGE, np.nan, values)
    n_after = np.isnan(values).sum()
    if n_after > n_before:
        print(f"[Note] Sanitized {n_after - n_before} out-of-range sentinel "
              f"values (|SPEI| > {SPEI_PLAUSIBLE_RANGE}) into proper NaN.")

    # Drop locations that are all-NaN (e.g. outside actual data coverage,
    # water bodies, or masked barren/sparse-vegetation areas per the paper)
    valid_mask = ~np.isnan(values).all(axis=0)
    values = values[:, valid_mask]
    n_valid = values.shape[1]
    print(f"Valid (non-NaN) grid cells: {n_valid}")

    if n_valid == 0:
        raise SystemExit(
            "No valid grid cells found -- the SPEI variable name, lat/lon "
            "names, or bounding box may need adjusting. Re-run with the "
            "printed dataset info above and adjust the script."
        )

    # Sample up to MAX_GRID_CELLS locations (mirrors "500 counties" sampling)
    rng = np.random.default_rng(SEED)
    n_sample = min(MAX_GRID_CELLS, n_valid)
    sample_idx = rng.choice(n_valid, size=n_sample, replace=False)
    values = values[:, sample_idx]
    print(f"Sampled {n_sample} grid cells (seed={SEED})")

    # Build sequences: for each location, slide a SEQ_LEN_MONTHS window
    # across its time series, predicting the class of the NEXT month.
    X, y = [], []
    for loc_i in range(values.shape[1]):
        series = values[:, loc_i]
        for t in range(SEQ_LEN_MONTHS, len(series)):
            window = series[t - SEQ_LEN_MONTHS:t]
            target = series[t]
            if np.isnan(window).any() or np.isnan(target):
                continue
            X.append(window)
            y.append(spei_to_class_4(target))

    X = np.array(X, dtype=np.float32)          # (N, SEQ_LEN_MONTHS)
    y = np.array(y, dtype=np.int64)             # (N,)
    print(f"\nBuilt {len(X)} sequences of length {SEQ_LEN_MONTHS} months")
    print(f"Class distribution: {np.bincount(y, minlength=4)} "
          f"({CLASS_LABELS_KZ})")

    if len(X) < 100:
        print("\n[WARNING] Very few sequences produced -- double check the "
              "bounding box and variable names above before proceeding.")

    # Reshape to match the US pipeline's (N, seq_len, n_features) convention,
    # with n_features=1 (SPEI only) instead of 18.
    X = X.reshape(len(X), SEQ_LEN_MONTHS, 1)

    # Chronological split (mirrors the US time-based train/val/test split)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    # Normalize (fit on train only, same approach as load_data.py)
    # IMPORTANT: use nanmean/nanstd as a safety net, and hard-assert no NaNs
    # remain afterward -- a single NaN here silently poisons the entire
    # training run (every loss becomes NaN, model learns nothing, and it
    # can look like it's "training" while actually doing nothing at all).
    mean = np.nanmean(X_train.reshape(-1, 1), axis=0).astype(np.float32)
    std  = np.nanstd(X_train.reshape(-1, 1), axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_test  = (X_test - mean) / std

    # Hard safety check -- if this fires, something upstream (the per-window
    # NaN filter) missed values, and we must NOT proceed to save a poisoned
    # dataset.
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            print(f"\n[ERROR] {name} still contains {n_nan} NaN values after "
                  f"normalization -- removing affected sequences before saving.")
            # Drop any sequence (row) that contains a NaN anywhere in its window
            if name == "X_train":
                good = ~np.isnan(X_train).any(axis=(1, 2))
                X_train, y_train, S_train = X_train[good], y_train[good], None
            elif name == "X_val":
                good = ~np.isnan(X_val).any(axis=(1, 2))
                X_val, y_val, S_val = X_val[good], y_val[good], None
            else:
                good = ~np.isnan(X_test).any(axis=(1, 2))
                X_test, y_test, S_test = X_test[good], y_test[good], None

    # Placeholder soil branch (all zeros) -- NO real Kazakhstan soil data
    # merged yet. This lets the EXISTING model class run unmodified, but
    # it means the soil branch contributes nothing meaningful here. This
    # is a clearly-flagged stand-in, not a real feature.
    soil_dim = 31   # match the US soil feature count so model.pt-style
                     # architectures can load without shape changes
    S_train = np.zeros((len(X_train), soil_dim), dtype=np.float32)
    S_val   = np.zeros((len(X_val), soil_dim), dtype=np.float32)
    S_test  = np.zeros((len(X_test), soil_dim), dtype=np.float32)

    # Final hard guarantee: refuse to save if any NaN survived the cleanup
    # above. Silently saving a poisoned file is worse than crashing here --
    # this is exactly what caused every training loss to become NaN last run.
    assert not np.isnan(X_train).any(), "X_train still has NaNs -- aborting save."
    assert not np.isnan(X_val).any(),   "X_val still has NaNs -- aborting save."
    assert not np.isnan(X_test).any(),  "X_test still has NaNs -- aborting save."
    print(f"\n[OK] Verified no NaN values remain in X_train/X_val/X_test.")

    np.savez(OUT_PATH,
             X_train=X_train, y_train=y_train, S_train=S_train,
             X_val=X_val,     y_val=y_val,     S_val=S_val,
             X_test=X_test,   y_test=y_test,   S_test=S_test,
             feature_names=np.array(["SPEI_6month"]),
             class_labels=np.array(CLASS_LABELS_KZ),
             means=mean, stds=std,
             seq_len_months=SEQ_LEN_MONTHS)

    print(f"\nSaved {OUT_PATH}")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Sequence shape: {X_train.shape} (SPEI-only, no soil data yet)")
    print(f"\nHONEST NOTE for presenting this: this uses REAL Kazakhstan-region")
    print(f"satellite/reanalysis-derived data (SPEI from CHIRPS+GLEAM), but")
    print(f"only 1 input variable (vs. 18 for the US model) and a zero-filled")
    print(f"soil placeholder (no real KZ soil data merged in yet). This proves")
    print(f"the PIPELINE and ARCHITECTURE transfer to real regional data --")
    print(f"it is a first step, not the full intended Kazakhstan study.")

if __name__ == "__main__":
    main()
