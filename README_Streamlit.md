# Streamlit Dashboard — Setup

## Install (in your existing `.venv`)
```bash
pip install streamlit
```

## Run
Copy `app.py` into your project folder (`F:\Demo_Satellite`), the same
folder that contains `results.json`, the 4 SHAP PNGs (`shap_summary.png`,
`shap_heatmap.png`, `shap_casestudy.png`, `shap_soil_importance.png`), and
`model.pt` + `sequences_{4,5,6}.npz` (for the Live Prediction tab).

Then:
```bash
streamlit run app.py
```
It opens automatically in your browser (usually http://localhost:8501).

## What's working

- **Overview** — project summary, run count from `results.json`
- **Model Comparison** — XGBoost vs LSTM-only vs LSTM+Attention, filterable
  by class scheme (4/5/6)
- **Class Scheme Comparison** — 4 vs 5 vs 6 class results side by side, with
  a confusion matrix viewer per scheme
- **SHAP Explainability** — 4 tabs: Global (weather) Importance, Feature ×
  Timestep Heatmap, Case Study, and Soil/Terrain Importance
- **Live Prediction** — pick a class scheme, click "Run prediction on a
  random test sample," and see:
  - True class vs. predicted class (with correct/incorrect indicator)
  - Prediction confidence per class (bar chart)
  - Attention weights across the 90-day window for that specific sample
    (line chart) — a live, per-sample version of the SHAP case study

## Important: Live Prediction requires a matching `model.pt`

`model.pt` only contains the weights for whichever `CLASS_SCHEME` you most
recently trained with `train_model.py`. If you select a different scheme in
the Live Prediction dropdown than what `model.pt` was actually trained on,
the app will show a clear error (not a crash) telling you to retrain with
the matching scheme first, or pick the scheme that matches your current
`model.pt`.

## Notes
- The dashboard reads `results.json` directly — every time you run
  `train_model.py` or `baselines.py`, just refresh the browser tab (or click
  Streamlit's "Rerun") to see the new numbers, no code changes needed.
- If a file (PNG, `results.json`, `sequences_N.npz`) is missing, that
  section shows a friendly warning instead of crashing — safe to open this
  at any point in your pipeline.
- Live Prediction requires `torch` to be installed in the same environment
  Streamlit is running in (it already is, if you're using the same `.venv`
  you trained with).
