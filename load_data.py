import numpy as np
import pandas as pd

DATA_DIR     = "./Data"
SEQ_LEN      = 90
CLASS_SCHEME = 6
N_COUNTIES   = 1000
SEED         = 42

assert CLASS_SCHEME in (4, 5, 6), "CLASS_SCHEME must be 4, 5, or 6"
N_CLASSES  = CLASS_SCHEME
OUT_PATH   = f"sequences_{CLASS_SCHEME}.npz"

FEATURE_COLS = [
    "PRECTOT","PS","QV2M","T2M","T2MDEW","T2MWET",
    "T2M_MAX","T2M_MIN","T2M_RANGE","TS",
    "WS10M","WS10M_MAX","WS10M_MIN","WS10M_RANGE",
    "WS50M","WS50M_MAX","WS50M_MIN","WS50M_RANGE",
]

# soil_data.csv columns other than "fips" (used as static per-county features)
SOIL_COLS = [
    "lat","lon","elevation",
    "slope1","slope2","slope3","slope4","slope5","slope6","slope7","slope8",
    "aspectN","aspectE","aspectS","aspectW","aspectUnknown",
    "WAT_LAND","NVG_LAND","URB_LAND","GRS_LAND","FOR_LAND",
    "CULTRF_LAND","CULTIR_LAND","CULT_LAND",
    "SQ1","SQ2","SQ3","SQ4","SQ5","SQ6","SQ7",
]

def score_to_class_6(s):
    if s < 1.0: return 0   # None
    elif s < 2.0: return 1  # D0
    elif s < 3.0: return 2  # D1
    elif s < 4.0: return 3  # D2
    elif s < 5.0: return 4  # D3
    else: return 5          # D4

def score_to_class_5(s):
    if s < 1.0: return 0   # None
    elif s < 2.0: return 1  # D0
    elif s < 3.0: return 2  # D1
    elif s < 4.0: return 3  # D2
    else: return 4          # D3+ (D3 and D4 merged)

def score_to_class_4(s):
    if s < 1.0: return 0   # None
    elif s < 2.0: return 1  # Mild
    elif s < 3.0: return 2  # Moderate
    else: return 3          # Severe+ (D2, D3, D4 merged)

SCORE_TO_CLASS = {4: score_to_class_4, 5: score_to_class_5, 6: score_to_class_6}
score_to_class = SCORE_TO_CLASS[CLASS_SCHEME]

CLASS_LABELS = {
    4: "0=None  1=Mild  2=Moderate  3=Severe+",
    5: "0=None  1=D0  2=D1  3=D2  4=D3+",
    6: "0=None  1=D0  2=D1  3=D2  4=D3  5=D4",
}

def get_counties():
    print("Scanning county list...")

    fips = pd.read_csv(f"{DATA_DIR}/train_timeseries/train_timeseries.csv",
                       usecols=["fips"])["fips"]
    all_fips = sorted(fips.unique())
    print(f"  Total counties: {len(all_fips)}")
    rng = np.random.default_rng(SEED)
    sampled = sorted(rng.choice(all_fips, size=N_COUNTIES, replace=False).tolist())
    print(f"  Sampled: {len(sampled)} (seed={SEED})")
    return sampled

def load_soil_data(fips_list):
    print(f"Loading {DATA_DIR}/soil_data.csv ...")
    soil = pd.read_csv(f"{DATA_DIR}/soil_data.csv")
    soil = soil[soil["fips"].isin(fips_list)].set_index("fips")
    missing = set(fips_list) - set(soil.index)
    if missing:
        print(f"  WARNING: {len(missing)} counties missing soil data, "
              f"will fill with column means: {sorted(missing)[:5]}...")
    # reindex to guarantee every sampled county has a row (fill missing w/ mean)
    soil = soil.reindex(fips_list)
    soil[SOIL_COLS] = soil[SOIL_COLS].fillna(soil[SOIL_COLS].mean())
    print(f"  Soil feature vector length: {len(SOIL_COLS)}")
    return soil[SOIL_COLS]

def build(name, fips_list, soil_df):
    path = f"{DATA_DIR}/{name}/{name}.csv"
    print(f"Loading {path} ...")
    df = pd.read_csv(path, parse_dates=["date"],
                     usecols=["fips","date","score"] + FEATURE_COLS)
    df = df[df["fips"].isin(fips_list)].reset_index(drop=True)
    print(f"  Rows: {len(df)}")
    X, y, S = [], [], []
    for fips_val, grp in df.groupby("fips"):
        grp = grp.sort_values("date").reset_index(drop=True)
        vals = grp[FEATURE_COLS].values.astype(np.float32)
        scrs = grp["score"].values
        soil_vec = soil_df.loc[fips_val].values.astype(np.float32)
        for i in range(SEQ_LEN, len(grp)):
            if pd.notna(scrs[i]) and not np.isnan(vals[i-SEQ_LEN:i]).any():
                X.append(vals[i-SEQ_LEN:i])
                y.append(score_to_class(scrs[i]))
                S.append(soil_vec)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    S = np.array(S, dtype=np.float32)
    print(f"  Sequences: {len(X)}, class dist: {np.bincount(y, minlength=N_CLASSES)}")
    return X, y, S

def main():
    fips = get_counties()
    soil_df = load_soil_data(fips)

    X_tr, y_tr, S_tr = build("train_timeseries", fips, soil_df)
    X_va, y_va, S_va = build("validation_timeseries", fips, soil_df)
    X_te, y_te, S_te = build("test_timeseries", fips, soil_df)

    flat  = X_tr.reshape(-1, X_tr.shape[-1])
    means = flat.mean(0).astype(np.float32)
    stds  = flat.std(0).astype(np.float32)
    stds[stds == 0] = 1.0

    X_tr = (X_tr - means) / stds
    X_va = (X_va - means) / stds
    X_te = (X_te - means) / stds

    # normalize static soil features the same way (fit on train only)
    soil_means = S_tr.mean(0).astype(np.float32)
    soil_stds  = S_tr.std(0).astype(np.float32)
    soil_stds[soil_stds == 0] = 1.0
    S_tr = (S_tr - soil_means) / soil_stds
    S_va = (S_va - soil_means) / soil_stds
    S_te = (S_te - soil_means) / soil_stds

    np.savez(OUT_PATH,
             X_train=X_tr, y_train=y_tr, S_train=S_tr,
             X_val=X_va,   y_val=y_va,   S_val=S_va,
             X_test=X_te,  y_test=y_te,  S_test=S_te,
             feature_names=np.array(FEATURE_COLS),
             soil_names=np.array(SOIL_COLS),
             means=means, stds=stds,
             soil_means=soil_means, soil_stds=soil_stds)

    print(f"\nSaved {OUT_PATH} (CLASS_SCHEME={CLASS_SCHEME}: {CLASS_LABELS[CLASS_SCHEME]})")
    print(f"  Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")
    print(f"  Sequence shape: {X_tr.shape} | Soil shape: {S_tr.shape}")

if __name__ == "__main__":
    main()
