import json
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Kazakhstan Task Studies", page_icon="\U0001F33F", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'Source Serif 4', serif !important; color: #2B2621 !important; letter-spacing: -0.01em; }
.stApp { background-color: #FAF6EE; }
section[data-testid="stSidebar"] { background-color: #F1E9D8; border-right: 1px solid #E4D9C4; }
section[data-testid="stSidebar"] h1 { font-size: 1.3rem !important; }
.field-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #B8592E; font-weight: 500; margin-bottom: -0.3rem; }
.kpi-card { background: #FFFFFF; border: 1px solid #E4D9C4; border-top: 4px solid var(--kpi-accent, #B8592E);
    border-radius: 6px; padding: 1rem 1.2rem; height: 100%; }
.kpi-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6B6258; }
.kpi-value { font-family: 'Source Serif 4', serif; font-size: 2.1rem; font-weight: 600; color: #2B2621; line-height: 1.2; }
.kpi-sub { font-size: 0.8rem; color: #6B6258; }
.field-divider { border: none; border-top: 1px solid #E4D9C4; margin: 1.5rem 0; }
.finding-card { background: #FFFFFF; border: 1px solid #E4D9C4; border-left: 4px solid #52736A;
    border-radius: 4px; padding: 0.9rem 1.1rem; margin-bottom: 0.7rem; }
.finding-card.warn { border-left-color: #8C3324; }
.finding-card.good { border-left-color: #B8592E; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

RESULTS_PATH = "results.json"
SUMMARY_PATH = "task_results_summary.json"
SHAP_FEATURE = "shap_kz_feature_importance.png"
SHAP_TIMESTEP = "shap_kz_timestep_importance.png"
SHAP_PERCLASS = "shap_kz_per_class.png"

KZ_FEATURES = "LSTM+Attention (Kazakhstan, engineered features)"
KZ_MULTISEED = "LSTM+Attention (Kazakhstan, engineered features, multi-seed)"
KZ_SCRATCH = "LSTM+Attention (KZ features, from scratch)"
KZ_TRANSFER = "LSTM+Attention (KZ features, transfer-learned)"
KZ_1RAW = "LSTM+Attention (Kazakhstan SPEI, real data)"
US_FULL = "LSTM+Attention (US, full 18-feat, arch-matched)"
US_PREC_SOIL = "LSTM+Attention (US, 1-feat PRECTOT + soil)"
US_PREC_ONLY = "LSTM+Attention (US, 1-feat PRECTOT)"
KZ_BINARY = "LSTM+Attention (KZ, binary stage 1)"
KZ_SEVERITY = "LSTM+Attention (KZ, severity stage 2)"
KZ_TWO_STAGE = "LSTM+Attention (KZ, two-stage combined)"
KZ_RESCUE = "LSTM+Attention (KZ, two-stage + rescue routing, seed42)"

CLASS_LABELS_KZ = ["None", "Mild", "Moderate", "Severe+"]


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


@st.cache_data
def load_summary():
    if not os.path.exists(SUMMARY_PATH):
        return {}
    with open(SUMMARY_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def run_entries(all_results, name):
    entries = all_results.get(name, [])
    if isinstance(entries, dict):
        entries = [entries]
    return entries


def latest_run(all_results, name):
    entries = run_entries(all_results, name)
    return entries[-1] if entries else None


def kpi_card(label, value, sub="", accent="#B8592E"):
    st.markdown(f"""<div class="kpi-card" style="--kpi-accent:{accent}">
        <div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)


def finding(text, kind=""):
    cls = f"finding-card {kind}" if kind else "finding-card"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def metrics_row(entry, key="Run"):
    return {
        key: entry.get("Run", entry.get("name", entry.get("variant", "-"))),
        "Accuracy": entry.get("Accuracy"),
        "Macro F1": entry.get("Macro F1"),
        "MAE": entry.get("MAE"),
        "RMSE": entry.get("RMSE"),
    }


st.sidebar.title("Kazakhstan Task Studies")
st.sidebar.markdown("AlfaFlow internship Tasks 1-5: feature engineering, "
                    "transfer learning, US control experiment, robustness, "
                    "two-stage pipeline and explainability -- on SPEI data.")
page = st.sidebar.radio("Navigate", [
    "Overview", "Task 1a \u2014 Feature Engineering", "Task 1b \u2014 Transfer Learning",
    "Task 1c \u2014 US Control Experiment", "Task 2 \u2014 Multi-seed Robustness",
    "Task 3 \u2014 Two-stage Pipeline", "Task 4 \u2014 Explainability",
    "Task 5 \u2014 Summary & Scope", "Controlled Experiments",
    "Live Prediction",
])

all_results = load_results()
summary = load_summary()


# --------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------- #
if page == "Overview":
    st.markdown('<div class="field-eyebrow">Overview</div>', unsafe_allow_html=True)
    st.title("Kazakhstan Drought Prediction \u2014 5 Task Studies")
    st.markdown("""Tasks 1(a,b,c) and 2-4 asked a set of sharp questions:
    does feature engineering help? does transfer learning from US data help?
    is the quality gap between US and Kazakhstan just about feature count?
    are the results stable across seeds? does a two-stage pipeline beat a
    single model? does the model reason the way meteorologists expect?""")

    st.subheader("Headline finding")
    kz_1 = latest_run(all_results, KZ_1RAW)
    us_1 = latest_run(all_results, US_PREC_ONLY)
    if kz_1 and us_1:
        gap = kz_1["Macro F1"] - us_1["Macro F1"]
        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Kazakhstan, 1 feature (SPEI)", f"{kz_1['Macro F1']:.3f}", "Macro F1", "#52736A")
        with c2: kpi_card("US, 1 feature (PRECTOT)", f"{us_1['Macro F1']:.3f}", "Macro F1", "#8C3324")
        with c3: kpi_card("Gap", f"+{gap:.3f}", "signal quality, not feature count", "#B8592E")
        finding("The quality difference is about <b>signal strength, not feature count</b>. "
                "One SPEI value (an index that integrates precipitation and evapotranspiration) "
                "beats one raw precipitation value by +0.196 Macro F1 \u2014 and even beats a "
                "full 18-feature US model.", kind="good")

    st.markdown('<hr class="field-divider">', unsafe_allow_html=True)
    st.subheader("Five tasks at a glance")
    best = latest_run(all_results, KZ_FEATURES)
    multiseed = run_entries(all_results, KZ_MULTISEED)
    scratch = latest_run(all_results, KZ_SCRATCH)
    transfer = latest_run(all_results, KZ_TRANSFER)
    twostage = latest_run(all_results, KZ_TWO_STAGE)

    spread = "-"
    if multiseed:
        f1s = [r["Macro F1"] for r in multiseed]
        spread = f"{max(f1s) - min(f1s):.4f}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Best Macro F1", f"{best['Macro F1']:.4f}" if best else "-",
                 "14-feature engineered model", "#52736A")
    with c2:
        kpi_card("Multi-seed spread", spread, "across 3 seeds", "#52736A")
    with c3:
        delta_feat = (best["Macro F1"] - kz_1["Macro F1"]) if (best and kz_1) else None
        kpi_card("Feature engineering", f"{delta_feat:+.3f}" if delta_feat is not None else "-",
                 "1-feature raw (0.542)", "#B8592E")
    with c4:
        delta_tl = (transfer["Macro F1"] - scratch["Macro F1"]) if (transfer and scratch) else None
        kpi_card("Transfer learning", f"{delta_tl:+.3f}" if delta_tl is not None else "-",
                 "vs from-scratch", "#8C3324")

    if twostage:
        kpi_card("Two-stage vs single", f"{twostage['Macro F1'] - best['Macro F1']:+.3f}"
                 if best else "-", "two-stage combined delta", "#8C3324")

    st.markdown('<hr class="field-divider">', unsafe_allow_html=True)
    st.subheader("The dataset")
    st.markdown("""
    - **SPEI-HR Central Asia** (Pyarali et al., *Scientific Data*, 2022) -- 5 km monthly
      drought index, 1981-2018, sourced from CHIRPS precipitation + GLEAM evapotranspiration
    - **Kazakhstan subset:** 500 grid cells sampled from the national bounding box
    - **Task 1a+ data:** 18-month lookback, 14 engineered SPEI features per month
      (\u2248149K sequences); original pipeline used 6-month lookback, 1 raw feature
    - **Classes:** 4-class severity scheme \u2014 None / Mild / Moderate / Severe+
    """)


# --------------------------------------------------------------------- #
# Task 1a
# --------------------------------------------------------------------- #
elif page == "Task 1a \u2014 Feature Engineering":
    st.markdown('<div class="field-eyebrow">Task 1a</div>', unsafe_allow_html=True)
    st.title("Feature Engineering")
    st.markdown("""**Question:** can richer features from the *same* SPEI series improve
    predictions \u2014 without any new data source?""")
    st.markdown("**Method:** built 14 features per month from raw SPEI: rolling min/max/std "
                "over 3/6/12-month windows, lags at 1/3/6 months, and rate of change. "
                "Extended the lookback window from 6 to 18 months.")

    kz_raw = latest_run(all_results, KZ_1RAW)
    kz_feat = latest_run(all_results, KZ_FEATURES)
    rows = []
    if kz_raw:
        rows.append({"Model": "1-feature raw SPEI", "Accuracy": kz_raw["Accuracy"],
                     "Macro F1": kz_raw["Macro F1"]})
    if kz_feat:
        rows.append({"Model": "14-feature engineered", "Accuracy": kz_feat["Accuracy"],
                     "Macro F1": kz_feat["Macro F1"]})
    if rows:
        df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(df.round(4), use_container_width=True)
        st.bar_chart(df[["Macro F1", "Accuracy"]])

    if kz_raw and kz_feat:
        d_f1 = kz_feat["Macro F1"] - kz_raw["Macro F1"]
        d_acc = kz_feat["Accuracy"] - kz_raw["Accuracy"]
        c1, c2 = st.columns(2)
        with c1: kpi_card("Macro F1 change", f"{d_f1:+.4f}", "0.5419 -> 0.6285", "#B8592E")
        with c2: kpi_card("Accuracy change", f"{d_acc:+.4f}", "0.8275 -> 0.8614", "#B8592E")
        finding("Feature engineering delivered the <b>largest single gain of the internship "
                f"({d_f1:+.3f} Macro F1)</b>. Rolling statistics, lags and rate-of-change extract "
                "temporal dynamics already present in SPEI \u2014 no new data was needed.", kind="good")

    st.subheader("The 14 engineered features")
    feats = kz_feat.get("feature_names") if kz_feat else []
    if feats:
        shown = {"raw": feats[:1], "rolling": feats[1:10], "lags": feats[10:13],
                 "rate": feats[13:]}
        st.markdown("**Raw:** " + ", ".join(shown["raw"]))
        st.markdown("**Rolling (min/max/std over 3/6/12 months):** " + ", ".join(shown["rolling"]))
        st.markdown("**Lags (1/3/6 months):** " + ", ".join(shown["lags"]))
        st.markdown("**Trend:** " + ", ".join(shown["rate"]))
    else:
        feats = ["spei_raw", "rolling_min_3m", "rolling_min_6m", "rolling_min_12m",
                 "rolling_max_3m", "rolling_max_6m", "rolling_max_12m",
                 "rolling_std_3m", "rolling_std_6m", "rolling_std_12m",
                 "lag_1m", "lag_3m", "lag_6m", "rate_of_change"]
        st.markdown("`" + "`, `".join(feats) + "`")

    st.markdown("The 18-month lookback needed to compute 12-month rolling statistics is "
                "stricter than the original 6-month pipeline, so the usable sequence count "
                "drops \u2014 yet F1 still rises, confirming the features carry real signal.")


# --------------------------------------------------------------------- #
# Task 1b
# --------------------------------------------------------------------- #
elif page == "Task 1b \u2014 Transfer Learning":
    st.markdown('<div class="field-eyebrow">Task 1b</div>', unsafe_allow_html=True)
    st.title("Transfer Learning from US Data")
    st.markdown("""**Question:** can weights pretrained on the much larger US drought
    dataset (874K sequences) help when fine-tuned on Kazakhstan (149K sequences)?""")
    st.markdown("""**Method:** trained a US 1-feature PRECTOT model (identical architecture,
    PROJ=24/HIDDEN=64), transferred 14 compatible layers to the Kazakhstan 14-feature model
    (all but the input projection, which has a shape mismatch 24x1 vs 24x14), and
    fine-tuned at a reduced learning rate (1e-4).""")

    scratch = latest_run(all_results, KZ_SCRATCH)
    transfer = latest_run(all_results, KZ_TRANSFER)
    rows = []
    if scratch:
        rows.append({"Model": "From scratch", "Accuracy": scratch["Accuracy"],
                     "Macro F1": scratch["Macro F1"]})
    if transfer:
        rows.append({"Model": "Transfer-learned (US weights)", "Accuracy": transfer["Accuracy"],
                     "Macro F1": transfer["Macro F1"]})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Model").round(4), use_container_width=True)
        st.bar_chart(pd.DataFrame(rows).set_index("Model")[["Macro F1", "Accuracy"]])

    if scratch and transfer:
        d = transfer["Macro F1"] - scratch["Macro F1"]
        c1, c2 = st.columns(2)
        with c1: kpi_card("Macro F1 change", f"{d:+.4f}", "transfer vs scratch", "#8C3324")
        with c2: kpi_card("Transferred layers", "14", "of 15; proj.0.weight excluded", "#52736A")
        finding("Transfer learning <b>hurt</b> performance. The pretrained weights were trained on "
                "PRECTOT (raw precipitation), while Kazakhstan trains on SPEI (a drought index). "
                "The two inputs encode <b>fundamentally different physical signals</b>, so the "
                "projection and LSTM representations from the US weights were actively "
                "incompatible with SPEI.", kind="warn")

    if transfer and "transferred_layers" in transfer:
        st.caption("Layers transferred: " + ", ".join(transfer["transferred_layers"]))

    st.markdown('<hr class="field-divider">', unsafe_allow_html=True)
    st.subheader("Spec-compliant re-run (US 18-feature base, only the input layer tuned)")
    st.markdown("The task asks literally to *\"use the weights of the model trained on US data "
                "(18 features) as a pretrained base, and fine-tune only the layer that sees the "
                "single available SPEI feature.\"* We re-ran the transfer to match that exactly "
                "on the 1-feature SPEI dataset, from the US 18-feature arch-matched model.")

    spec_scratch = latest_run(all_results, "LSTM+Attention (KZ, 1-feat, from scratch 24/64)")
    spec_frozen = latest_run(all_results, "LSTM+Attention (KZ, 1-feat, US18 transfer frozen)")
    spec_fullft = latest_run(all_results, "LSTM+Attention (KZ, 1-feat, US18 transfer full-finetune)")
    rows = []
    if spec_scratch:
        rows.append({"Model": "From scratch", "Trainable params": "30,100",
                     "Accuracy": spec_scratch["Accuracy"], "Macro F1": spec_scratch["Macro F1"]})
    if spec_frozen:
        rows.append({"Model": "US18 frozen + input layer (spec)", "Trainable params": "48",
                     "Accuracy": spec_frozen["Accuracy"], "Macro F1": spec_frozen["Macro F1"]})
    if spec_fullft:
        rows.append({"Model": "US18 full fine-tune", "Trainable params": "30,100",
                     "Accuracy": spec_fullft["Accuracy"], "Macro F1": spec_fullft["Macro F1"]})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Model").round(4), use_container_width=True)
        st.bar_chart(pd.DataFrame([{"Model": r["Model"], "Macro F1": r["Macro F1"]}
                                   for r in rows]).set_index("Model"))
        finding("The literal spec recipe \u2014 freezing the entire US tower and training only the "
                "48-parameter input projection \u2014 performs <b>worst (F1\u22480.32)</b>: a frozen "
                "feature extractor trained on US precipitation cannot be reshaped by a tiny input "
                "adapter into a useful SPEI representation. Full fine-tune (0.508) is better but "
                "still below training from scratch (0.542). <b>Whatever the transfer recipe, US "
                "weights do not compensate for the missing features.</b>", kind="warn")


# --------------------------------------------------------------------- #
# Task 1c
# --------------------------------------------------------------------- #
elif page == "Task 1c \u2014 US Control Experiment":
    st.markdown('<div class="field-eyebrow">Task 1c</div>', unsafe_allow_html=True)
    st.title("US Control Experiment")
    st.markdown("""**Question:** if the US had as *few* features as Kazakhstan, would the
    quality gap disappear? Is the gap about feature count \u2014 or about the signal itself?""")
    st.markdown("""**Method:** trained three US variants with the architecture matched to the
    Kazakhstan model (PROJ=24/HIDDEN=64): full 18-feature, 1-feature PRECTOT + soil,
    and 1-feature PRECTOT only (the closest analog to SPEI).""")

    us_full = latest_run(all_results, US_FULL)
    us_ps = latest_run(all_results, US_PREC_SOIL)
    us_p = latest_run(all_results, US_PREC_ONLY)
    kz_raw = latest_run(all_results, KZ_1RAW)
    kz_feat = latest_run(all_results, KZ_FEATURES)

    rows = []
    for entry, label in [(us_full, "US full (18-feat + soil)"),
                         (us_ps, "US PRECTOT + soil"),
                         (us_p, "US PRECTOT only (1 feat)"),
                         (kz_raw, "Kazakhstan SPEI (1 feat)"),
                         (kz_feat, "Kazakhstan SPEI (14 engineered)")]:
        if entry:
            rows.append({"Model": label, "Accuracy": entry["Accuracy"], "Macro F1": entry["Macro F1"]})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.set_index("Model").round(4), use_container_width=True)
        st.bar_chart(df.set_index("Model")[["Macro F1", "Accuracy"]])

    st.markdown("**... plus the honest caveat:** the US models here were trained on a 200,000-row "
                "subsample of the full 874K-row dataset to fit on a laptop GPU. The ranking is the "
                "same as on the full dataset \u2014 only the absolute level shifts slightly.")

    if kz_raw and us_p:
        gap = kz_raw["Macro F1"] - us_p["Macro F1"]
        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("US 1-feature (PRECTOT)", f"{us_p['Macro F1']:.3f}", "Macro F1", "#8C3324")
        with c2: kpi_card("KZ 1-feature (SPEI)", f"{kz_raw['Macro F1']:.3f}", "Macro F1", "#52736A")
        with c3: kpi_card("Gap", f"+{gap:.3f}", "identical feature count", "#B8592E")
        finding("Single-feature comparison: <b>0.346 (US) vs 0.542 (KZ) \u2014 a +0.196 gap</b>. "
                "SPEI is a far stronger single drought signal than raw precipitation. Even adding "
                "soil data only lifts the US PRECTOT model to 0.369 \u2014 still far below Kazakhstan's "
                "1-feature SPEI. <b>The quality gap is about the signal, not the feature count.</b>",
                kind="good")
        finding("Because SPEI already integrates precipitation and evapotranspiration into one "
                "statistically standardized drought index, one SPEI feature carries more "
                "drought-relevant signal than 18 raw weather variables the US model must learn "
                "to interpret from scratch.")


# --------------------------------------------------------------------- #
# Task 2
# --------------------------------------------------------------------- #
elif page == "Task 2 \u2014 Multi-seed Robustness":
    st.markdown('<div class="field-eyebrow">Task 2</div>', unsafe_allow_html=True)
    st.title("Multi-seed Robustness")
    st.markdown("""**Question:** is the reported 0.6285 F1 reliable \u2014 or an artifact of
    one lucky random seed?""")
    st.markdown("""**Method:** retrained the 14-feature Kazakhstan model on seeds 42, 123 and
    2026 with identical hyperparameters (PROJ=24/HIDDEN=64, batch 64, LR 5e-4, patience 30).""")

    multiseed = run_entries(all_results, KZ_MULTISEED)
    if multiseed:
        rows = [{"Seed": r.get("seed"), "Macro F1": r["Macro F1"], "Accuracy": r["Accuracy"],
                 "MAE": r.get("MAE"), "RMSE": r.get("RMSE")} for r in multiseed]
        df = pd.DataFrame(rows)
        st.dataframe(df.set_index("Seed").round(4), use_container_width=True)
        st.bar_chart(df.set_index("Seed")[["Macro F1"]])

        stats = summary.get("tasks", {}).get("task_2_multiseed", {}).get("statistics", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Mean Macro F1", f"{stats.get('macro_f1_mean', 0.0):.4f}",
                          "3 seeds", "#52736A")
        with c2: kpi_card("Spread", f"{stats.get('macro_f1_spread', 0.0):.4f}",
                          "max - min", "#52736A")
        with c3: kpi_card("Std (sample)", f"{stats.get('macro_f1_std', 0.0):.4f}",
                          "ddof=1", "#52736A")
        with c4: kpi_card("Range", "-".join(f"{v:.3f}" for v in stats.get("macro_f1_range", [])),
                          "min ... max", "#52736A")

        finding("Results are <b>stable</b>: Macro F1 spread 0.0078 is well below the 0.03 "
                "reliability threshold. The model's performance is not attributable to a lucky "
                "initialization.", kind="good")


# --------------------------------------------------------------------- #
# Task 3
# --------------------------------------------------------------------- #
elif page == "Task 3 \u2014 Two-stage Pipeline":
    st.markdown('<div class="field-eyebrow">Task 3</div>', unsafe_allow_html=True)
    st.title("Two-stage Pipeline")
    st.markdown("""**Question:** does a staged pipeline \u2014 a binary drought detector followed
    by a severity classifier \u2014 beat a single 4-class model?""")
    st.markdown("""**Method:** Stage 1 classifies None vs drought; Stage 2, run only on samples
    flagged as drought, grades Mild/Moderate/Severe+. Combined output: Stage 1 decides if there
    is drought, Stage 2 assigns severity.""")

    single = latest_run(all_results, KZ_FEATURES)
    binary = latest_run(all_results, KZ_BINARY)
    severity = latest_run(all_results, KZ_SEVERITY)
    combined = latest_run(all_results, KZ_TWO_STAGE)
    rescue = latest_run(all_results, KZ_RESCUE)

    rows = []
    if single:
        rows.append({"Stage": "Single-stage 4-class (reference)", "Accuracy": single["Accuracy"],
                     "Macro F1": single["Macro F1"]})
    if binary:
        rows.append({"Stage": "Stage 1: binary detector", "Accuracy": binary["Accuracy"],
                     "Macro F1": binary["Macro F1"]})
    if severity:
        rows.append({"Stage": "Stage 2: severity classifier", "Accuracy": severity["Accuracy"],
                     "Macro F1": severity["Macro F1"]})
    if combined:
        rows.append({"Stage": "Two-stage combined (hard gate)", "Accuracy": combined["Accuracy"],
                     "Macro F1": combined["Macro F1"]})
    if rescue:
        rows.append({"Stage": "Two-stage + rescue routing", "Accuracy": rescue["Accuracy"],
                     "Macro F1": rescue["Macro F1"]})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Stage").round(4), use_container_width=True)
        st.bar_chart(pd.DataFrame(rows).set_index("Stage")[["Macro F1"]])

    if single and combined:
        d = combined["Macro F1"] - single["Macro F1"]
        c1, c2 = st.columns(2)
        with c1: kpi_card("Two-stage combined (hard gate)", f"{combined['Macro F1']:.4f}",
                          "corrected assembly", "#B8592E")
        with c2: kpi_card("Delta vs single-stage", f"{d:+.4f}", "Macro F1", "#52736A")
        if rescue:
            dr = rescue["Macro F1"] - single["Macro F1"]
            c3, c4 = st.columns(2)
            with c3: kpi_card("Two-stage + rescue routing", f"{rescue['Macro F1']:.4f}",
                              "ambiguous band \u2192 4-class model", "#52736A")
            with c4: kpi_card("Delta vs single-stage", f"{dr:+.4f}", "Macro F1", "#52736A")
        finding("Earlier reports showed a -0.401 two-stage 'collapse'; that was an <b>evaluation bug</b> "
                "(batch-local indices written into the global test array in `train_two_stage.py`), now "
                "fixed and re-run on all 3 seeds. Corrected: the hard-gate two-stage pipeline scores "
                "~0.61 Macro F1 (only ~0.02 below single-stage); three-way rescue routing reaches ~0.63 "
                "\u2014 statistically tied with the single-stage model.", kind="good")
        finding("Single-stage remains the simplest, lowest-variance design and is recommended. The "
                "two-stage pipeline is viable but delivers no quality gain. The real, smaller caveat "
                "persists: Stage 1 flags ~20% of droughts as no-drought under a hard gate; rescue "
                "routing is what recovers those samples.")
    if combined and "binary_f1" in combined:
        st.markdown("**Cascade in numbers:** binary detector accuracy "
                    f"{combined.get('binary_acc', 0):.4f}, binary drought recall ~0.80; severity "
                    f"stage accuracy {combined.get('severity_acc', 0):.4f}. Even a strong stage 1 "
                    "leaks errors into stage 2.")


# --------------------------------------------------------------------- #
# Task 4
# --------------------------------------------------------------------- #
elif page == "Task 4 \u2014 Explainability":
    st.markdown('<div class="field-eyebrow">Task 4</div>', unsafe_allow_html=True)
    st.title("Explainability (Permutation Importance)")
    st.markdown("""**Method:** on the best model (seed 42, F1=0.6285), permute each feature /
    timestep independently on the test set and measure the Macro F1 drop (3 repeats). A larger
    drop = more important.""")

    top3 = [
        ("spei_raw", "the current drought state is by far the strongest signal"),
        ("rate_of_change", "the model tracks the drought trajectory, not just the level"),
        ("lag_6m", "six-month-lagged SPEI gives meaningful secondary signal"),
    ]
    for name, desc in top3:
        finding(f"<b>{name}</b> \u2014 {desc}")

    st.subheader("Charts")
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(SHAP_FEATURE):
            st.image(SHAP_FEATURE, use_container_width=True)
            st.caption("Feature importance: Macro F1 drop when each feature is permuted.")
        else:
            st.warning(f"`{SHAP_FEATURE}` not found.")
    with c2:
        if os.path.exists(SHAP_TIMESTEP):
            st.image(SHAP_TIMESTEP, use_container_width=True)
            st.caption("Timestep importance across the 18-month window.")
        else:
            st.warning(f"`{SHAP_TIMESTEP}` not found.")
    if os.path.exists(SHAP_PERCLASS):
        st.image(SHAP_PERCLASS, use_container_width=True)
        st.caption("Per-class feature importance heatmap.")
    else:
        st.warning(f"`{SHAP_PERCLASS}` not found.")

    task4 = summary.get("tasks", {}).get("task_4_explainability", {})
    if task4:
        st.subheader("Top features")
        fr = task4.get("feature_importance_ranked", [])
        if fr:
            st.dataframe(pd.DataFrame([{"Feature": r["feature"], "Macro F1 drop": r["f1_drop"]}
                                       for r in fr]).set_index("Feature").round(4),
                         use_container_width=True)
        tr = task4.get("timestep_importance_ranked", [])
        if tr:
            st.markdown("**Recent months dominate:** the three most recent months (16-18) and the "
                        "lag months carry most of the predictive weight. This matches the "
                        "meteorological expectation that the near-term drought state drives the "
                        "next month's outcome.")


# --------------------------------------------------------------------- #
# Task 5
# --------------------------------------------------------------------- #
elif page == "Task 5 \u2014 Summary & Scope":
    st.markdown('<div class="field-eyebrow">Task 5</div>', unsafe_allow_html=True)
    st.title("Summary & Scope")
    st.markdown("The full written report is in `REPORT_Task_5_Internship.md` with the machine-readable "
                "numbers in `task_results_summary.json`.")

    st.subheader("Key decisions")
    conclusion = summary.get("final_conclusions", {})
    if conclusion:
        gains = [("Feature engineering from SPEI", conclusion.get("biggest_quality_gain", {}).get("impact", "-")),
                 ("Using SPEI instead of PRECTOT", conclusion.get("second_gain", {}).get("impact", "-"))]
        c1, c2 = st.columns(2)
        for (title, val), col in zip(gains, [c1, c2]):
            with col: kpi_card(title, val, "Macro F1", "#52736A")
        finding(conclusion.get("robustness", ""), kind="good")

    st.subheader("Failed approaches")
    for fa in conclusion.get("failed_approaches", []):
        finding(f"<b>{fa.get('approach')}</b> \u2014 {fa.get('impact')}. {fa.get('reason')}",
                kind="warn")

    st.markdown('<hr class="field-divider">', unsafe_allow_html=True)

    st.subheader("Done since the report draft")
    st.markdown("""
    - **Multi-seed robustness for transfer learning and two-stage models** is now done:
      the transfer penalty (scratch F1 0.616-0.629 vs transferred 0.523-0.541, mean
      **-0.09**) holds across seeds 42/123/2026.
    - **Two-stage evaluation corrected:** the earlier -0.40 result was a batch-index
      bug in the evaluation script (now fixed and documented). Corrected hard-gate
      two-stage is **~0.61** Macro F1, and rescue routing **~0.63** \u2014 a statistical
      tie with single-stage 0.63, not a failure.
    """)

    st.subheader("Out of scope per the internship spec (future work)")
    st.markdown("""
    The internship task is restricted to open SPEI-HR Central Asia data and explicitly
    does not require searching for unavailable data; the missing soil data is called in
    the spec an *objective data-availability limitation, not a technical shortcoming*.
    These items are intentionally excluded and listed for transparency:
    - Real Kazakhstan soil-moisture data (the soil branch is a zero-filled placeholder)
    - More weather variables (temperature, humidity, wind, ...) as inputs
    - A CNN branch for satellite imagery (the current pipeline is sequence-only;
      imagery is weeks of future work)
    - Validation across more than 500 grid cells (~14% of Kazakhstan's 5 km grid)
    - Uncertainty quantification for operational deployment
    """)


# --------------------------------------------------------------------- #
# Live Prediction (stretch goal -- kept only if it loads cleanly)
# --------------------------------------------------------------------- #
elif page == "Live Prediction":
    st.markdown('<div class="field-eyebrow">Interactive demo</div>', unsafe_allow_html=True)
    st.title("Live Prediction \u2014 Kazakhstan")
    st.markdown("Load the trained 14-feature model and run a prediction on a random test sample.")

    DATA_PATH = "sequences_kazakhstan_features.npz"
    WEIGHTS = {
        "model_kazakhstan_features.pt (seed 42, best)": "model_kazakhstan_features.pt",
        "model_kazakhstan_features_seed42.pt": "model_kazakhstan_features_seed42.pt",
        "model_kazakhstan_features_seed123.pt": "model_kazakhstan_features_seed123.pt",
        "model_kazakhstan_features_seed2026.pt": "model_kazakhstan_features_seed2026.pt",
    }

    if not os.path.exists(DATA_PATH):
        st.warning(f"`{DATA_PATH}` not found \u2014 run `load_data_kazakhstan_features.py` first.")
        st.stop()
    available = {k: v for k, v in WEIGHTS.items() if os.path.exists(v)}
    if not available:
        st.warning("No trained weight files found. Run `train_model_kazakhstan_features.py` "
                   "or the multi-seed script first.")
        st.stop()

    weight_label = st.selectbox("Weights", list(available.keys()))
    weight_path = available[weight_label]

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        st.error("`torch` is not installed in this Streamlit environment.")
        st.stop()

    PROJ_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ = 24, 64, 1, 0.4, 8

    class LSTMAttention(nn.Module):
        def __init__(self, input_size, soil_size, n_classes):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(input_size, PROJ_SIZE), nn.LayerNorm(PROJ_SIZE),
                                      nn.ReLU(), nn.Dropout(DROPOUT * 0.5))
            self.lstm = nn.LSTM(PROJ_SIZE, HIDDEN_SIZE, N_LAYERS, batch_first=True)
            self.attn_w = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
            self.attn_v = nn.Linear(HIDDEN_SIZE, 1, bias=False)
            self.drop = nn.Dropout(DROPOUT)
            self.soil_proj = nn.Sequential(nn.Linear(soil_size, SOIL_PROJ), nn.LayerNorm(SOIL_PROJ),
                                           nn.ReLU(), nn.Dropout(DROPOUT * 0.5))
            self.fc1 = nn.Linear(HIDDEN_SIZE + SOIL_PROJ, 32)
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
            return self.fc2(self.relu(self.fc1(combined))), weights

    @st.cache_resource
    def load_model_and_data(data_path, weight_path):
        data = np.load(data_path, allow_pickle=True)
        X_test, S_test, y_test = data["X_test"], data["S_test"], data["y_test"]
        model = LSTMAttention(X_test.shape[-1], S_test.shape[-1], 4)
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        labels = list(data.get("class_labels", np.array(CLASS_LABELS_KZ)))
        return model, X_test, S_test, y_test, labels

    try:
        model, X_test, S_test, y_test, labels = load_model_and_data(DATA_PATH, weight_path)
    except RuntimeError as e:
        st.error(f"Couldn't load these weights with the 14-feature architecture.\n\n{e}")
        st.stop()

    st.caption("Note: `S_test` is a zero-filled placeholder \u2014 no real Kazakhstan soil data "
               "is used (see Task 5 scope).")

    if "kz_live_pred_idx" not in st.session_state:
        st.session_state.kz_live_pred_idx = None
    if st.button("Run prediction on a random test sample"):
        rng = np.random.default_rng()
        st.session_state.kz_live_pred_idx = int(rng.integers(0, len(X_test)))

    if st.session_state.kz_live_pred_idx is not None:
        idx = st.session_state.kz_live_pred_idx
        x = torch.tensor(X_test[idx:idx + 1], dtype=torch.float32)
        s = torch.tensor(S_test[idx:idx + 1], dtype=torch.float32)
        true_label = int(y_test[idx])
        with torch.no_grad():
            logits, attn_weights = model(x, s)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_label = int(probs.argmax())
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("True class", labels[true_label])
        with col2: st.metric("Predicted class", labels[pred_label],
                             delta="Correct" if pred_label == true_label else "Incorrect",
                             delta_color="normal" if pred_label == true_label else "inverse")
        with col3: st.metric("Confidence", f"{probs[pred_label]:.2%}")
        st.subheader("Prediction confidence")
        st.bar_chart(pd.DataFrame({"Class": labels, "Probability": probs}).set_index("Class"))
        st.subheader("Attention weights over the 18-month window")
        attn = attn_weights.numpy()[0]
        st.line_chart(pd.DataFrame({"Month": np.arange(1, len(attn) + 1),
                                    "Attention weight": attn}).set_index("Month"))


# --------------------------------------------------------------------- #
# Controlled Experiments
# --------------------------------------------------------------------- #
elif page == "Controlled Experiments":
    st.markdown('<div class="field-eyebrow">Supplementary</div>', unsafe_allow_html=True)
    st.title("Controlled Scenarios \u2014 C/D Comparison & Validated Two-stage")
    st.markdown("Same architecture, same cells (seed 42, 500 locations), same training recipe. "
                "Built on frozen spec modules (`model_lstm_attention.py`, `metrics_task_spec.py`). "
                "Per-location chronological 70/15/15 split. Seeds 42 / 123 / 2026.")

    # --- Task 1 C/D ---
    st.subheader("Task 1: Scenario C vs D")
    t1_path = "task1_results.json"
    if os.path.exists(t1_path):
        with open(t1_path) as f:
            t1 = json.load(f)
        summary_t1 = t1.get("summary_by_scenario", {})

        c1, c2, c3 = st.columns(3)
        for col, sc, label, accent in [
            (c1, "C", "Scenario C (raw SPEI 6\u00d71)", "#52736A"),
            (c2, "D", "Scenario D (engineered 6\u00d722)", "#B8592E"),
            (c3, None, "Delta", "#2B2621"),
        ]:
            with col:
                m = summary_t1.get(sc, {}).get("Macro F1", {}) if sc else {}
                a = summary_t1.get(sc, {}).get("Accuracy", {}) if sc else {}
                if sc:
                    kpi_card(label, f"{m.get('mean', 0):.4f}",
                             f"\u00b1{m.get('std', 0):.4f}  |  Acc {a.get('mean', 0):.4f}", accent)
                else:
                    delta = (summary_t1.get("D", {}).get("Macro F1", {}).get("mean", 0)
                             - summary_t1.get("C", {}).get("Macro F1", {}).get("mean", 0))
                    kpi_card(label, f"{delta:+.4f}", "C \u2192 D Macro F1 change", accent)

        runs_rows = []
        for sc in ("C", "D"):
            for run in t1["individual_runs"].get(sc, []):
                runs_rows.append({"Scenario": sc, "Seed": run["seed"],
                                  "Epochs": run.get("epochs", "-"),
                                  "Accuracy": run.get("Accuracy"),
                                  "Macro F1": run.get("Macro F1"),
                                  "MAE": run.get("MAE"),
                                  "Severe+ FNR": run.get("Severe+ False Negative Rate")})
        st.dataframe(pd.DataFrame(runs_rows).set_index(["Scenario", "Seed"]), use_container_width=True)

        finding("Engineered features (D) do **not** beat raw SPEI (C) on 6-month windows "
                "\u2014 delta **\u22120.0074** with ~8\u00d7 higher seed variance. "
                "SPEI\u2019s raw value carries most of the signal.", kind="warn")
    else:
        st.warning(f"`{t1_path}` not found \u2014 run `run_task1_controlled.py` first.")

    st.markdown('<hr class="field-divider">', unsafe_allow_html=True)

    # --- Task 2 two-stage ---
    st.subheader("Task 2: Validated Two-stage Pipeline (Scenario D)")
    t2_path = "task2_results.json"
    if os.path.exists(t2_path):
        with open(t2_path) as f:
            t2 = json.load(f)
        t2s = t2.get("summary", {})
        sev_errs = t2s.get("critical_errors", {})

        c1, c2, c3 = st.columns(3)
        for col, label, key, accent in [
            (c1, "Single 4-class", "single_stage", "#52736A"),
            (c2, "Two-stage combined", "two_stage_combined", "#B8592E"),
            (c3, "Accuracy gain", None, "#2B2621"),
        ]:
            with col:
                if key:
                    m = t2s.get(key, {}).get("Macro F1", {})
                    a = t2s.get(key, {}).get("Accuracy", {})
                    kpi_card(label, f"{m.get('mean', 0):.4f}",
                             f"\u00b1{m.get('std', 0):.4f}  |  Acc {a.get('mean', 0):.4f}", accent)
                else:
                    acc_d = (t2s.get("two_stage_combined", {}).get("Accuracy", {}).get("mean", 0)
                             - t2s.get("single_stage", {}).get("Accuracy", {}).get("mean", 0))
                    kpi_card(label, f"+{acc_d:.4f}", "Combined vs single-stage Accuracy", accent)

        t2_runs = pd.DataFrame([
            {"Seed": r["seed"],
             "Single MF1": r["single"]["Macro F1"],
             "Combined MF1": r["combined"]["Macro F1"],
             "Single Acc": r["single"]["Accuracy"],
             "Combined Acc": r["combined"]["Accuracy"],
             "Flagged": r["combined"].get("confusion_matrix", [[0]])}
            for r in t2["runs"]
        ])
        display_runs = t2_runs[["Seed", "Single MF1", "Combined MF1", "Single Acc", "Combined Acc"]].copy()
        st.dataframe(display_runs.set_index("Seed"), use_container_width=True)

        st.subheader("Combined confusion matrix (seed 42)")
        cm42 = t2["runs"][0]["combined"]["confusion_matrix"]
        cm_df = pd.DataFrame(cm42, index=["None", "Mild", "Mod", "Sev+"],
                             columns=["None", "Mild", "Mod", "Sev+"])
        st.dataframe(cm_df, use_container_width=True)

        st.subheader("Critical errors (mean over seeds)")
        err_df = pd.DataFrame([
            {"Error type": k, "Rate": f"{sev_errs.get(k, {}).get('mean', 0):.1%}"}
            for k in ["Severe+ predicted as None rate", "Moderate predicted as Mild rate",
                       "Mild predicted as Severe+ rate", "None predicted as Drought rate"]
        ])
        st.dataframe(err_df.set_index("Error type"), use_container_width=True)

        finding("Two-stage combined ties single-stage on Macro F1 (+0.005, inside seed noise) "
                "but wins on **Accuracy (+0.011)** with **3\u00d7 lower seed variance**. "
                "Routing uses stage-1 **predictions**, not ground truth "
                "(an oracle-gate defect was caught and fixed).", kind="good")
    else:
        st.warning(f"`{t2_path}` not found \u2014 run `run_task2_two_stage.py` first.")