import json
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Drought Field Report", page_icon="\U0001F326", layout="wide")

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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

RESULTS_PATH = "results.json"
SHAP_SUMMARY, SHAP_HEATMAP = "shap_summary.png", "shap_heatmap.png"
SHAP_CASESTUDY, SHAP_SOIL = "shap_casestudy.png", "shap_soil_importance.png"

CLASS_LABELS = {
    4: ["None", "Mild", "Moderate", "Severe+"],
    5: ["None", "D0", "D1", "D2", "D3+"],
    6: ["None", "D0", "D1", "D2", "D3", "D4"],
}
NO_NONE_LABELS = {4: ["Mild", "Moderate", "Severe+"]}

@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def get_runs(all_results, model_name):
    entries = all_results.get(model_name, [])
    if isinstance(entries, dict):
        entries = [entries]
    return entries

def latest_run_per_scheme(all_results, model_name):
    out = {}
    for r in get_runs(all_results, model_name):
        nc = r.get("n_classes")
        if nc is not None:
            out[nc] = r
    return out

def kpi_card(label, value, sub="", accent="#B8592E"):
    st.markdown(f"""<div class="kpi-card" style="--kpi-accent:{accent}">
        <div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

st.sidebar.title("\U0001F326 Drought Field Report")
st.sidebar.markdown("LSTM + Attention model for US county-level drought severity, "
                     "with SHAP explainability and soil/terrain features.")
page = st.sidebar.radio("Navigate", [
    "Overview", "Field Report (walkthrough)", "Model Comparison",
    "Class Scheme Comparison", "Class Imbalance Experiments",
    "Kazakhstan Portability", "SHAP Explainability", "Live Prediction",
    "Kazakhstan Task Studies",
])

all_results = load_results()

if page == "Overview":
    st.markdown('<div class="field-eyebrow">Project overview</div>', unsafe_allow_html=True)
    st.title("Drought Severity Prediction")
    st.markdown("""
    **Task:** predict US county-level drought severity from the last 90 days
    of meteorological data plus static soil/terrain features.

    **Model:** LSTM + Attention over the daily weather sequence, combined
    with a small MLP branch for static soil features, feeding a final
    classification head.
    """)
    if all_results:
        latest4 = latest_run_per_scheme(all_results, "LSTM+Attention (ours)").get(4)
        n_runs = sum(len(get_runs(all_results, k)) for k in all_results)
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Total logged runs", str(n_runs), "across all models & schemes", "#B8592E")
        with c2:
            if latest4: kpi_card("Best Macro F1 (4-class)", f"{latest4['Macro F1']:.3f}",
                                  "LSTM+Attention, latest run", "#52736A")
        with c3: kpi_card("Imbalance techniques tried", "4", "weights / focal \u00d73 gammas / None-exclusion", "#B8592E")
        with c4: kpi_card("Soil features", "31", "elevation, slope, land use, soil quality", "#52736A")
        st.markdown('<hr class="field-divider">', unsafe_allow_html=True)
    st.markdown("**Pipeline:**")
    st.markdown("""
    1. `load_data.py` \u2014 builds 90-day sequences + merges soil data, saves `sequences_N.npz`
    2. `train_model.py` \u2014 trains the LSTM+Attention model (class weights + Focal Loss)
    3. `train_no_none.py` \u2014 severity-only variant, "None" class excluded
    4. `baselines.py` \u2014 trains XGBoost + LSTM-only for comparison
    5. `explain.py` \u2014 SHAP feature importance analysis
    """)
    if not all_results:
        st.warning("No `results.json` found yet in this folder.")

# --------------------------------------------------------------------- #
elif page == "Field Report (walkthrough)":
    st.title("Field Report")
    st.caption("A step-by-step narrative of the project -- data, model, and what we found.")
    step = st.radio("Step", ["1. The data", "2. The model", "3. Label granularity",
                              "4. Soil data", "5. Class imbalance", "6. What we learned"],
                     horizontal=True, label_visibility="collapsed")
    st.markdown('<hr class="field-divider">', unsafe_allow_html=True)

    if step == "1. The data":
        st.subheader("US Drought Meteorological Data")
        st.markdown("""
        - **Source:** NASA POWER reanalysis + US Drought Monitor scores + Harmonized World Soil Database
        - **Coverage:** 3,108 US counties, daily, 2000-2020
        - **This project uses:** 500-1,000 randomly sampled counties, due to memory constraints
        - **Input:** 90-day rolling window of 18 meteorological variables per county
        """)

    elif step == "2. The model":
        st.subheader("LSTM + Attention, with a soil/terrain branch")
        st.code("""
Branch 1 -- Weather sequence: (batch, 90, 18)
  -> Linear(18->32) + LayerNorm + ReLU + Dropout   [Input Projection]
  -> LSTM(32->96, 1 layer)                          [Temporal Encoding]
  -> Additive Attention over 90 timesteps           [Attention]
  -> weighted sum -> 96-dim context vector

Branch 2 -- Static soil/terrain: (batch, 31)
  -> Linear(31->16) + LayerNorm + ReLU + Dropout     [Soil Projection]

Combine: concatenate [weather context (96) | soil vector (16)]
  -> Linear(112->32) + ReLU -> Linear(32->N_CLASSES)
        """, language="text")

    elif step == "3. Label granularity":
        st.subheader("Why three class schemes?")
        latest_by_scheme = latest_run_per_scheme(all_results, "LSTM+Attention (ours)")
        rows = [{"Classes": nc, "Macro F1": r["Macro F1"], "Accuracy": r["Accuracy"]}
                for nc in [4, 5, 6] if (r := latest_by_scheme.get(nc))]
        if rows:
            st.line_chart(pd.DataFrame(rows).set_index("Classes")[["Macro F1", "Accuracy"]])
        st.markdown('<div class="finding-card">Macro F1 declines as class count increases -- '
                    'driven by data scarcity in the rarest classes (D3, D4), not an architecture flaw.</div>',
                    unsafe_allow_html=True)

    elif step == "4. Soil data":
        st.subheader("Adding static soil/terrain features")
        col1, col2 = st.columns(2)
        with col1: kpi_card("Without soil data", "0.302", "Macro F1, 6-class", "#8C3324")
        with col2: kpi_card("With soil data", "0.331", "Macro F1, 6-class", "#52736A")
        st.markdown('<div class="finding-card">The model\'s top soil predictor is <b>longitude</b> -- '
                    'more important than any single weather variable, acting as a proxy for US climate zone.</div>',
                    unsafe_allow_html=True)
        if os.path.exists(SHAP_SOIL): st.image(SHAP_SOIL, use_container_width=True)

    elif step == "5. Class imbalance":
        st.subheader("Solving class imbalance -- tested, then re-tested")
        st.markdown("""
        Our supervisor raised two direct questions: **(1) solve the class
        imbalance problem**, and **(2) is this compatible with Kazakhstan?**
        This step covers imbalance; the next step covers findings overall.
        """)
        st.markdown("**A. Class weights** (baseline, always on) -- upweight rare classes in the loss.")
        st.markdown("**B. Focal Loss** -- additionally down-weights *easy* predictions, "
                     "forcing the model to focus on hard/rare examples. Initially tested at "
                     "3 strengths (single seed each), then re-tested across 3 seeds per "
                     "strength (9 runs total) to check reliability:")

        gamma_summary = [
            {"Gamma": 0.5, "Mean Macro F1": 0.4485, "Mean Severe+ Recall": 0.685},
            {"Gamma": 1.0, "Mean Macro F1": 0.4629, "Mean Severe+ Recall": 0.563},
            {"Gamma": 2.0, "Mean Macro F1": 0.4631, "Mean Severe+ Recall": 0.626},
        ]
        st.dataframe(pd.DataFrame(gamma_summary).set_index("Gamma"), use_container_width=True)
        st.markdown(
            '<div class="finding-card warn">A single-seed run initially suggested gamma=1.0 was '
            'clearly best. Averaged across 3 seeds, that advantage disappeared -- '
            'seed-to-seed variance turned out to be larger than the difference between gamma '
            'values. No gamma showed a reliable, repeatable advantage. This taught us to '
            'distrust single-run comparisons going forward.</div>', unsafe_allow_html=True)

        st.markdown("**C. Excluding \"None\" entirely** (supervisor's direct question) -- "
                     "train only on Mild/Moderate/Severe+, removing the dominant class altogether:")
        none_rows = [
            {"Setting": "Full scheme (with None)", "Macro F1": 0.4646, "Accuracy": 0.7750},
            {"Setting": "None excluded", "Macro F1": 0.5110, "Accuracy": 0.5814},
        ]
        st.dataframe(pd.DataFrame(none_rows).set_index("Setting"), use_container_width=True)
        st.markdown(
            '<div class="finding-card">Excluding "None" improves Macro F1 (0.465\u21920.511) and '
            'Mild-class recall (33%\u219272%) -- confirming the intuition, in this single run. '
            'But accuracy drops (no more "easy" None class to lean on), and this model alone can '
            'no longer tell whether a county is drought-free at all. Given what the gamma sweep '
            'taught us, this specific result should also be re-checked across multiple seeds '
            'before treating it as fully confirmed.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="finding-card warn">Practical implication: a "None-excluded" model would need '
            'to be paired with a separate first-stage binary (drought / no-drought) classifier to be '
            'deployable -- this experiment tests the idea in isolation, not a full two-stage system.</div>',
            unsafe_allow_html=True)

    elif step == "6. What we learned":
        st.subheader("Key findings")
        st.markdown('<div class="finding-card">LSTM+Attention outperformed both baselines '
                    '(XGBoost, LSTM-only) on Accuracy and Macro F1 across nearly every configuration tested.</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="finding-card">Fewer classes = higher Macro F1, explained by data '
                    'scarcity in D3/D4, not a modeling weakness.</div>', unsafe_allow_html=True)
        st.markdown('<div class="finding-card warn">Scaling 500\u21921,000 counties did NOT improve '
                    'results -- geographic/climate diversity is a harder generalization problem than '
                    'data volume alone solves.</div>', unsafe_allow_html=True)
        st.markdown('<div class="finding-card">Soil/terrain features add real signal -- longitude alone '
                    'outranks every weather variable.</div>', unsafe_allow_html=True)
        st.markdown('<div class="finding-card">Class imbalance: tested 4 approaches (weights, 3 Focal '
                    'gammas, None-exclusion). Focal \u03b3=1.0 gave the best Severe+ recall (69.3%); '
                    'excluding None gave the best overall Macro F1 (0.511) but needs a two-stage '
                    'system to be deployable.</div>', unsafe_allow_html=True)
        st.markdown("**Next steps:** full 3,108-county training, then apply this pipeline to "
                     "Kazakhstan satellite data (MODIS NDVI, CHIRPS rainfall, ERA5-Land, GRACE) -- "
                     "requires a CNN branch for imagery and a new label scheme (SPEI-3).")

# --------------------------------------------------------------------- #
elif page == "Model Comparison":
    st.markdown('<div class="field-eyebrow">Benchmarking</div>', unsafe_allow_html=True)
    st.title("Model Comparison")
    st.caption("US drought models only. Kazakhstan portability results are on their own page.")
    COMPARISON_MODELS = ["XGBoost (baseline)", "LSTM-only (baseline)", "LSTM+Attention (ours)"]
    if not all_results:
        st.warning("No results found.")
    else:
        # Only scan the THREE actual comparison models for available class
        # schemes -- scanning every key in results.json (including the
        # Kazakhstan experiments, which use n_classes=3 for their no-None
        # variant) would surface schemes with no matching baseline runs.
        all_schemes = sorted({r["n_classes"] for m in COMPARISON_MODELS
                               for r in get_runs(all_results, m) if r.get("n_classes") is not None})
        if all_schemes:
            scheme = st.selectbox("Class scheme", all_schemes,
                                   format_func=lambda n: f"{n}-class ({', '.join(CLASS_LABELS.get(n, []))})")
            rows = []
            for model_name in ["XGBoost (baseline)", "LSTM-only (baseline)", "LSTM+Attention (ours)"]:

                latest = latest_run_per_scheme(all_results, model_name).get(scheme)
                if latest:
                    rows.append({"Model": model_name, "Accuracy": latest.get("Accuracy"),
                                 "Macro F1": latest.get("Macro F1"), "MAE": latest.get("MAE"),
                                 "RMSE": latest.get("RMSE"), "Uses soil data": latest.get("uses_soil_data", False),
                                 "Run timestamp": latest.get("timestamp", "-")})
            if rows:
                df = pd.DataFrame(rows).set_index("Model")
                st.dataframe(df.round(4), use_container_width=True)
                st.bar_chart(df[["Accuracy", "Macro F1"]])
            else:
                st.info(f"No runs found yet for the {scheme}-class scheme.")

# --------------------------------------------------------------------- #
elif page == "Class Scheme Comparison":
    st.markdown('<div class="field-eyebrow">Label granularity</div>', unsafe_allow_html=True)
    st.title("Class Scheme Comparison (LSTM+Attention)")
    st.markdown("Comparing **4-class**, **5-class**, and **6-class** granularity "
                "\u2014 same model, same data, different target granularity.")
    latest_by_scheme = latest_run_per_scheme(all_results, "LSTM+Attention (ours)")
    if not latest_by_scheme:
        st.warning("No LSTM+Attention runs found yet.")
    else:
        rows = [{"Classes": f"{nc} ({', '.join(CLASS_LABELS.get(nc, []))})", "n_classes": nc,
                 "Accuracy": r.get("Accuracy"), "Macro F1": r.get("Macro F1"),
                 "MAE": r.get("MAE"), "RMSE": r.get("RMSE")}
                for nc in sorted(latest_by_scheme) for r in [latest_by_scheme[nc]]]
        df = pd.DataFrame(rows)
        st.dataframe(df.set_index("Classes").round(4), use_container_width=True)
        st.line_chart(df.set_index("n_classes")[["Macro F1", "Accuracy"]])
        st.subheader("Confusion Matrix")
        scheme_pick = st.selectbox("View confusion matrix for", sorted(latest_by_scheme),
                                    format_func=lambda n: f"{n}-class", key="cm_scheme_pick")
        cm = latest_by_scheme[scheme_pick].get("confusion_matrix")
        if cm:
            labels = CLASS_LABELS.get(scheme_pick, [str(i) for i in range(scheme_pick)])
            st.caption("Rows = true class, columns = predicted class")
            st.dataframe(pd.DataFrame(cm, index=labels, columns=labels), use_container_width=True)

# --------------------------------------------------------------------- #
elif page == "Class Imbalance Experiments":
    st.title("Class Imbalance Experiments")
    st.markdown("""
    Direct response to: *"1st) solve class imbalance problem."* Multiple
    approaches were tested, layered on top of each other -- including a
    multi-seed check that **revised our initial conclusion**.
    """)

    st.subheader("A + B. Class weights + Focal Loss (gamma \u00d7 seed sweep)")
    st.caption("9 runs: 3 gamma values \u00d7 3 random seeds (42, 123, 2026), same "
               "data and config otherwise -- to separate a real gamma effect "
               "from ordinary run-to-run noise.")

    sweep_rows = [
        {"Gamma": 0.5, "Seed": 42,   "Macro F1": 0.4486, "Severe+ Recall": 0.685},
        {"Gamma": 0.5, "Seed": 123,  "Macro F1": 0.4533, "Severe+ Recall": 0.729},
        {"Gamma": 0.5, "Seed": 2026, "Macro F1": 0.4436, "Severe+ Recall": 0.641},
        {"Gamma": 1.0, "Seed": 42,   "Macro F1": 0.4664, "Severe+ Recall": 0.459},
        {"Gamma": 1.0, "Seed": 123,  "Macro F1": 0.4744, "Severe+ Recall": 0.598},
        {"Gamma": 1.0, "Seed": 2026, "Macro F1": 0.4478, "Severe+ Recall": 0.632},
        {"Gamma": 2.0, "Seed": 42,   "Macro F1": 0.4672, "Severe+ Recall": 0.530},
        {"Gamma": 2.0, "Seed": 123,  "Macro F1": 0.4700, "Severe+ Recall": 0.696},
        {"Gamma": 2.0, "Seed": 2026, "Macro F1": 0.4522, "Severe+ Recall": 0.652},
    ]
    df_sweep = pd.DataFrame(sweep_rows)
    st.dataframe(df_sweep.set_index(["Gamma", "Seed"]).round(4), use_container_width=True)

    st.markdown("**Averaged across seeds:**")
    summary = df_sweep.groupby("Gamma").agg(
        Mean_Macro_F1=("Macro F1", "mean"),
        Mean_Severe_Recall=("Severe+ Recall", "mean"),
        Min_Severe_Recall=("Severe+ Recall", "min"),
        Max_Severe_Recall=("Severe+ Recall", "max"),
    ).round(4)
    st.dataframe(summary, use_container_width=True)
    st.bar_chart(df_sweep.groupby("Gamma")[["Severe+ Recall", "Macro F1"]].mean())

    st.markdown(
        '<div class="finding-card warn"><b>Revised finding:</b> a single-seed run initially '
        'suggested gamma=1.0 was clearly best (69.3% Severe+ recall). Testing across 3 seeds '
        'showed gamma=1.0\'s Severe+ recall actually ranges from 45.9% to 63.2% depending on '
        'seed alone -- that single result was a lucky run, not a reliable gamma effect. '
        'Averaged properly, gamma=0.5 has the highest mean Severe+ recall (68.5%), but all '
        'three gammas overlap heavily -- <b>seed-to-seed variance is larger than the '
        'difference between gamma values</b>. We do not have strong evidence that any one '
        'gamma is reliably better than the others.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="finding-card">This is itself a useful methodological finding: it shows '
        'why single-run comparisons can be misleading, and why we re-tested before reporting '
        'a firm conclusion.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="field-divider">', unsafe_allow_html=True)

    st.subheader('C. Excluding "None" entirely (severity-only model)')
    st.caption('Direct test of: "can we exclude the None class and do the remaining task?"')
    none_rows = [
        {"Setting": "Full scheme (with None)", "Macro F1": 0.4646, "Accuracy": 0.7750,
         "Mild Recall": 0.327, "Moderate Recall": 0.178, "Severe+ Recall": 0.693},
        {"Setting": "None excluded", "Macro F1": 0.5110, "Accuracy": 0.5814,
         "Mild Recall": 0.719, "Moderate Recall": 0.309, "Severe+ Recall": 0.559},
    ]
    df_none = pd.DataFrame(none_rows).set_index("Setting")
    st.dataframe(df_none.round(4), use_container_width=True)

    st.markdown("**Does this hold up across ALL models, or just ours?** "
                "We also trained XGBoost and LSTM-only on the same None-excluded data:")
    none_all_models = [
        {"Model": "XGBoost", "With None (F1)": 0.4129, "None excluded (F1)": 0.4534, "Change": "+0.041"},
        {"Model": "LSTM-only", "With None (F1)": 0.4564, "None excluded (F1)": 0.4900, "Change": "+0.034"},
        {"Model": "LSTM+Attention (ours)", "With None (F1)": 0.4646, "None excluded (F1)": 0.5110, "Change": "+0.046"},
    ]
    st.dataframe(pd.DataFrame(none_all_models).set_index("Model"), use_container_width=True)
    st.markdown(
        '<div class="finding-card">Removing "None" improved Macro F1 for <b>all three models</b>, '
        'not just ours -- confirming this is a genuine, general effect of the class imbalance, '
        'not an artifact specific to our architecture. LSTM+Attention still wins every metric in '
        'the None-excluded setting, and gained the most from the change (+0.046), suggesting the '
        'attention mechanism makes best use of the added focus once "None" no longer dominates.</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="finding-card warn">Trade-off: overall accuracy drops sharply (77.5%\u219258.1%), '
        'since there\'s no "easy" None class to boost the score, and this model CANNOT determine '
        'whether a county is in drought at all -- it only ranks severity among counties already '
        'known to be in drought. Deploying this would require a separate first-stage binary '
        '(drought / no-drought) classifier ahead of it.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="field-divider">', unsafe_allow_html=True)
    st.subheader("Summary: what to report")
    st.markdown("""
    - **Focal Loss gamma:** no gamma value showed a reliable advantage once tested
      across multiple seeds -- the effect we initially saw was noise, not signal.
      Class weights alone remain a solid, simple baseline; Focal Loss at any tested
      gamma is roughly equivalent to it.
    - **Excluding "None":** shows a real, directionally consistent improvement in
      Macro F1 and Mild-class recall in the run we have, but hasn't yet been
      re-validated across seeds the way the gamma sweep was -- flagged as a
      promising but not fully confirmed result.
    - **Practical implication:** class weighting (already in the baseline model) is
      doing most of the real work; further gains likely need a different lever
      entirely (e.g. oversampling, or the two-stage drought/severity approach),
      not more Focal Loss tuning.
    """)

# --------------------------------------------------------------------- #
elif page == "Kazakhstan Portability":
    st.title("Kazakhstan Portability")
    st.markdown("""
    Direct response to: *"2nd) is it compatible to use in Kazakhstan?"*
    We tested this with **real data**, not a synthetic simulation.
    """)

    st.subheader("The dataset")
    st.markdown("""
    **SPEI-HR Central Asia** (Pyarali et al., *Scientific Data*, 2022) --
    a real, published, open-access (CC BY 4.0) dataset covering Kazakhstan
    and the wider Central Asia region.
    - 5km spatial resolution, monthly, 1981-2018
    - Built from CHIRPS precipitation + GLEAM evapotranspiration
    - Uses **SPEI** -- exactly the drought index our project already
      targets for the full Kazakhstan study
    - Source: CEDA Archive (catalogue.ceda.ac.uk)
    """)

    st.subheader("What we built")
    st.markdown("""
    Adapted the full US pipeline to this data: cropped to Kazakhstan's
    bounding box, sampled 500 grid cells (mirroring the "500 counties"
    approach), built 6-month rolling windows (SPEI's native timescale),
    classified into the same 4-class drought scheme, and trained the
    **same LSTM+Attention architecture** (scaled down: 1 input feature
    instead of 18, no real soil data yet -- placeholder only).
    """)

    st.markdown(
        '<div class="finding-card warn">Along the way we found and fixed a real data-quality '
        'bug: the NetCDF file stored missing-data pixels as an extreme sentinel value '
        '(a huge out-of-range number) rather than a proper NaN, which silently poisoned '
        'training (loss went to NaN, model collapsed to predicting one class). Fixed by '
        'explicitly sanitizing out-of-range values before any NaN-based filtering.</div>',
        unsafe_allow_html=True)

    st.subheader("Results")
    kz_rows = [
        {"Setting": "US (4-class, 500 counties)", "Macro F1": 0.4718, "Accuracy": 0.7778,
         "Input features": 18, "Soil data": "Yes (31 features)"},
        {"Setting": "Kazakhstan (4-class, with None)", "Macro F1": 0.5419, "Accuracy": 0.8275,
         "Input features": 1, "Soil data": "No (placeholder)"},
        {"Setting": "Kazakhstan (None excluded)", "Macro F1": 0.5596, "Accuracy": 0.6738,
         "Input features": 1, "Soil data": "No (placeholder)"},
    ]
    st.dataframe(pd.DataFrame(kz_rows).set_index("Setting"), use_container_width=True)

    st.markdown(
        '<div class="finding-card">Kazakhstan Macro F1 (0.542) is <b>higher</b> than our US '
        'result (0.472), despite having only 1 input feature versus 18 and no real soil data. '
        'Likely explanation: SPEI is already a purpose-built, statistically standardized drought '
        'index -- it carries more concentrated signal per feature than raw weather variables the '
        'US model has to learn drought-relevance for from scratch.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="finding-card">Excluding "None" on the Kazakhstan data shows the same '
        'trade-off pattern as the US experiment (higher F1, lower accuracy), but the effect is '
        'smaller in magnitude here (+0.018 vs +0.046 F1) -- consistent with SPEI already being a '
        'cleaner signal even with None included.</div>', unsafe_allow_html=True)

    st.subheader("What's still simplified -- honest scope")
    st.markdown("""
    - **1 input feature** (SPEI) vs. 18 for the US model -- no temperature,
      humidity, wind, etc. yet
    - **No real Kazakhstan soil/terrain data** -- the soil branch is a
      zero-filled placeholder, not a real Kazakhstan-specific source
    - **Approximate bounding box** -- not validated against an official
      Kazakhstan administrative boundary shapefile
    - **Smaller model** (fewer hidden units) to match the smaller,
      simpler dataset -- not a straight architecture transplant
    """)
    st.markdown("**This proves the pipeline and architecture transfer to real "
                 "regional data.** It is a first step, not the finished Kazakhstan "
                 "study -- enriching it with more variables, real soil data, and "
                 "eventually satellite imagery (requiring a CNN branch) is realistically "
                 "weeks of further work, not a quick addition.")

# --------------------------------------------------------------------- #
elif page == "SHAP Explainability":
    st.markdown('<div class="field-eyebrow">Explainability</div>', unsafe_allow_html=True)
    st.title("SHAP Explainability")
    st.markdown("Feature importance analysis from `explain.py`.")
    tabs = st.tabs(["Global Importance", "Feature \u00d7 Timestep Heatmap", "Case Study", "Soil/Terrain Importance"])
    with tabs[0]:
        if os.path.exists(SHAP_SUMMARY): st.image(SHAP_SUMMARY, use_container_width=True)
        else: st.warning(f"`{SHAP_SUMMARY}` not found.")
    with tabs[1]:
        if os.path.exists(SHAP_HEATMAP): st.image(SHAP_HEATMAP, use_container_width=True)
        else: st.warning(f"`{SHAP_HEATMAP}` not found.")
    with tabs[2]:
        if os.path.exists(SHAP_CASESTUDY): st.image(SHAP_CASESTUDY, use_container_width=True)
        else: st.warning(f"`{SHAP_CASESTUDY}` not found.")
    with tabs[3]:
        if os.path.exists(SHAP_SOIL):
            st.image(SHAP_SOIL, use_container_width=True)
            st.caption("Static per-county soil/terrain features -- shown as a single importance "
                       "ranking since they don't vary day-to-day.")
        else:
            st.warning(f"`{SHAP_SOIL}` not found.")

# --------------------------------------------------------------------- #
elif page == "Live Prediction":
    st.markdown('<div class="field-eyebrow">Interactive demo</div>', unsafe_allow_html=True)
    st.title("Live Prediction")
    st.markdown("Pick a class scheme, then run the trained model on a random test-set sample.")
    scheme = st.selectbox("Class scheme to load", [4, 5, 6], index=0)
    data_path = f"sequences_{scheme}.npz"
    st.code(f"{data_path}  +  model.pt  (model.pt must have been trained with CLASS_SCHEME={scheme})")

    if not os.path.exists(data_path):
        st.warning(f"`{data_path}` not found. Run `load_data.py` with CLASS_SCHEME={scheme} first.")
    elif not os.path.exists("model.pt"):
        st.warning("`model.pt` not found. Run `train_model.py` first.")
    else:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            st.error("`torch` is not installed in this Streamlit environment.")
            st.stop()

        PROJ_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ = 32, 96, 1, 0.4, 16

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
                B, T, F = x.shape
                x = self.proj(x.reshape(B * T, F)).reshape(B, T, -1)
                out, _ = self.lstm(x)
                w = torch.softmax(self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1), dim=1)
                ctx = (out * w.unsqueeze(-1)).sum(dim=1)
                soil_ctx = self.soil_proj(s)
                combined = torch.cat([self.drop(ctx), soil_ctx], dim=1)
                return self.fc2(self.relu(self.fc1(combined))), w

        @st.cache_resource
        def load_model_and_data(data_path, n_classes):
            data = np.load(data_path, allow_pickle=True)
            X_test, S_test, y_test = data["X_test"], data["S_test"], data["y_test"]
            model = LSTMAttention(X_test.shape[-1], S_test.shape[-1], n_classes)
            state = torch.load("model.pt", map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            return model, X_test, S_test, y_test

        try:
            model, X_test, S_test, y_test = load_model_and_data(data_path, scheme)
        except RuntimeError as e:
            st.error(f"Couldn't load `model.pt` for a {scheme}-class model -- mismatch with "
                     f"how model.pt was trained.\n\nDetails: {e}")
            st.stop()

        if "live_pred_idx" not in st.session_state:
            st.session_state.live_pred_idx = None
        if st.button("Run prediction on a random test sample"):
            st.session_state.live_pred_idx = int(np.random.default_rng().integers(0, len(X_test)))

        if st.session_state.live_pred_idx is not None:
            idx = st.session_state.live_pred_idx
            x = torch.tensor(X_test[idx:idx+1], dtype=torch.float32)
            s = torch.tensor(S_test[idx:idx+1], dtype=torch.float32)
            true_label = int(y_test[idx])
            with torch.no_grad():
                logits, attn_weights = model(x, s)
                probs = torch.softmax(logits, dim=1).numpy()[0]
                pred_label = int(probs.argmax())
            labels = CLASS_LABELS.get(scheme, [str(i) for i in range(scheme)])
            col1, col2 = st.columns(2)
            with col1: st.metric("True class", labels[true_label])
            with col2: st.metric("Predicted class", labels[pred_label],
                                 delta="Correct" if pred_label == true_label else "Incorrect",
                                 delta_color="normal" if pred_label == true_label else "inverse")
            st.subheader("Prediction confidence")
            st.bar_chart(pd.DataFrame({"Class": labels, "Probability": probs}).set_index("Class"))
            st.subheader("Attention weights over the 90-day window")
            attn = attn_weights.numpy()[0]
            st.line_chart(pd.DataFrame({"Day": np.arange(1, len(attn)+1),
                                        "Attention weight": attn}).set_index("Day"))

# --------------------------------------------------------------------- #
elif page == "Kazakhstan Task Studies":
    st.markdown('<div class="field-eyebrow">Companion dashboard</div>', unsafe_allow_html=True)
    st.title("Kazakhstan Task Studies")
    st.markdown("The internship deliverables for Kazakhstan (Tasks 1-5) -- feature "
                "engineering, transfer learning, the US control experiment, multi-seed "
                "robustness, the two-stage pipeline, and explainability -- live in a "
                "separate dashboard that shares this project's data and styling.")
    st.markdown("**Launch it in a second terminal:**")
    st.code("streamlit run app_kazakhstan_5tasks.py", language="bash")
    st.markdown("""It reads the same `results.json` plus the curated
    `task_results_summary.json`, and renders the Task 4 explainability charts -- kept
    separate from this US-focused report so each app stays focused.""")
