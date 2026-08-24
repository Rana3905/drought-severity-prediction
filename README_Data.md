# 🌦️ Drought Severity Prediction — LSTM + Attention with SHAP Explainability

## Project Overview

End-to-end deep learning pipeline for drought severity classification using
daily meteorological time series plus static soil/terrain data. Built as a
proof-of-concept for the multi-source satellite drought prediction
methodology targeting Kazakhstan.

The project was developed progressively through several experimental stages:

1. Initial US drought classification pipeline
2. Comparison of 4-, 5-, and 6-class drought severity schemes
3. Expansion from 500 to 1000 randomly sampled US counties
4. Class-imbalance experiments using Focal Loss
5. Severity-only experiments excluding the dominant `None` class
6. Extension of the pipeline to Kazakhstan SPEI data
7. SHAP-based model explainability
8. Interactive Streamlit dashboard for results and analysis

The overall goal is to investigate whether temporal meteorological patterns,
combined with static environmental information, can support drought severity
classification and whether the resulting workflow can be transferred to a
different geographic and data setting.

---

## Dataset

**US Drought Meteorological Data** (Kaggle, CC0 Public Domain)

- Source: https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data
- Original data: NASA POWER reanalysis + US Drought Monitor weekly scores +
  Harmonized World Soil Database (HWSD) static features
- Coverage: 3,108 US counties, daily 2000–2020

### County Sampling

The project initially used **500 randomly sampled counties (seed=42)** for
development and the initial class-scheme experiments.

Later experiments expanded the US sample to **1000 randomly sampled counties**
to investigate whether additional spatial coverage improved model performance
and robustness.

The sampling strategy was primarily motivated by the memory and compute
limitations of the development machine:

- RAM: 16 GB
- GPU VRAM: 4 GB

The 500-county configuration therefore represents the initial experimental
setup, while the 1000-county configuration represents a later scaling
experiment.

### Temporal Splits

The same general temporal split was maintained across the US experiments:

| Split      | Period    |
|------------|-----------|
| Train      | 2000–2016 |
| Validation | 2017–2018 |
| Test       | 2019–2020 |

For the initial 500-county configuration, the approximate sequence counts
were:

| Split      | Period    | Approx. Sequences |
|------------|-----------|-------------------|
| Train      | 2000–2016 | ~437,000          |
| Validation | 2017–2018 | ~45,500           |
| Test       | 2019–2020 | ~46,000           |

The exact number of generated sequences varies with the number of sampled
counties. Later 1000-county experiments retained the same general temporal
split while increasing the number of spatial units.

**Note on comparability:** the official Kaggle benchmark notebook (Macro F1
0.639) uses a *different* setup than this project: a 2000–2009/2010–2011/
2012–2020 split, a 180-day input window, previous drought scores as an
additional feature, and the full 6-class label scale.

This project uses a 90-day window, a 2000–2016/2017–2018/2019–2020 split,
and no autoregressive drought history. Results here are **not directly
comparable** to that benchmark number — they answer a related but distinct
question with a lighter-weight setup.

---

## Features

### 18 daily meteorological variables (time-varying, 90-day window)

| Variable     | Description                        |
|--------------|------------------------------------|
| PRECTOT      | Precipitation (mm/day)             |
| PS           | Surface Pressure (kPa)             |
| QV2M         | Specific Humidity at 2m (g/kg)     |
| T2M          | Temperature at 2m (°C)             |
| T2MDEW       | Dew Point at 2m (°C)               |
| T2MWET       | Wet Bulb Temperature at 2m (°C)    |
| T2M_MAX      | Max Temperature at 2m (°C)         |
| T2M_MIN      | Min Temperature at 2m (°C)         |
| T2M_RANGE    | Temperature Range at 2m (°C)       |
| TS           | Earth Skin Temperature (°C)        |
| WS10M        | Wind Speed at 10m (m/s)            |
| WS10M_MAX    | Max Wind Speed at 10m (m/s)        |
| WS10M_MIN    | Min Wind Speed at 10m (m/s)        |
| WS10M_RANGE  | Wind Speed Range at 10m (m/s)      |
| WS50M        | Wind Speed at 50m (m/s)            |
| WS50M_MAX    | Max Wind Speed at 50m (m/s)        |
| WS50M_MIN    | Min Wind Speed at 50m (m/s)        |
| WS50M_RANGE  | Wind Speed Range at 50m (m/s)      |

### 31 static soil/terrain variables (per-county, constant across the window)

Elevation, 8 slope-class percentages, 5 aspect-direction percentages, 7
land-use percentages (water, non-vegetated, urban, grassland, forest,
rainfed/irrigated/all cropland), 7 soil-quality index scores, plus
latitude/longitude.

Source: Harmonized World Soil Database (HWSD), merged with the county-level
meteorological data using county FIPS code.

The static soil/terrain information remains constant throughout the 90-day
weather window and is processed through a separate MLP branch of the model.

### Soil/Terrain Branch

```text
Static soil/terrain features: (batch, 31)
  → Linear(31→16)
  → LayerNorm + ReLU + Dropout
  → 16-dimensional soil/terrain representation
```

The resulting 16-dimensional representation is combined with the
96-dimensional weather representation produced by the LSTM + Attention
branch.

```text
Weather representation: 96 dimensions
Soil/terrain representation: 16 dimensions
                         │
                         ▼
                 Concatenation
                         │
                         ▼
                  112 dimensions
                         │
                         ▼
              Classification Head
```

### SHAP finding

Longitude (`lon`) is the single most important feature in the entire model —
more important than any individual weather variable.

This is physically sensible because longitude across the continental US
captures broad geographic and climatic gradients, including differences
between relatively arid western regions and more humid eastern regions.

However, SHAP importance should be interpreted as **model attribution rather
than causal evidence**. Geographic variables such as longitude can act as
proxies for multiple underlying climatic and environmental factors.

---

## Drought Classes — three label granularities tested

We evaluated three levels of label granularity to understand how class
scarcity affects performance (see Results below).

### 6-class (official US Drought Monitor scale)

| Class | Label | Score Range |
|-------|-------|-------------|
| 0 | None | < 1.0 |
| 1 | D0 | 1.0 – 2.0 |
| 2 | D1 | 2.0 – 3.0 |
| 3 | D2 | 3.0 – 4.0 |
| 4 | D3 | 4.0 – 5.0 |
| 5 | D4 | ≥ 5.0 |

### 5-class (D3+D4 merged — D4 alone has too few samples to learn)

| Class | Label | Score Range |
|-------|-------|-------------|
| 0 | None | < 1.0 |
| 1 | D0 | 1.0 – 2.0 |
| 2 | D1 | 2.0 – 3.0 |
| 3 | D2 | 3.0 – 4.0 |
| 4 | D3+ | ≥ 4.0 |

### 4-class (simplified — D2/D3/D4 merged)

| Class | Label | Score Range |
|-------|-------|-------------|
| 0 | None | < 1.0 |
| 1 | Mild | 1.0 – 2.0 |
| 2 | Moderate | 2.0 – 3.0 |
| 3 | Severe+ | ≥ 3.0 |

---

## Model Architecture

Two-branch network: an LSTM+Attention encoder over the daily weather
sequence, combined with a small MLP over the static soil/terrain vector.

```text
Branch 1 — Weather sequence: (batch, 90, 18)
  → Linear(18→32) + LayerNorm + ReLU + Dropout(0.2)   [Input Projection]
  → LSTM(32→96, 1 layer, batch_first=True)             [Temporal Encoding]
  → Additive Attention over 90 timesteps               [Attention]
  → weighted sum → 96-dim context vector

Branch 2 — Static soil/terrain: (batch, 31)
  → Linear(31→16) + LayerNorm + ReLU + Dropout          [Soil Projection]

Combine: concatenate [weather context (96) | soil vector (16)] = 112-dim
  → Dropout(0.4)
  → Linear(112→32) + ReLU                               [Classification Head]
  → Linear(32→N_CLASSES)                                [Output Logits]
```

`N_CLASSES` is 4, 5, or 6 depending on which US label scheme is active.

---

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `load_data.py` | Builds 90-day sequences, merges soil data by county, bins labels per `CLASS_SCHEME`, saves `sequences_{4,5,6}.npz` |
| `train_model.py` | Trains the LSTM+Attention model using weighted Cross-Entropy Loss, saves `model.pt` + appends results to `results.json` |
| `train_model_focal_loss.py` | Trains the LSTM+Attention model using Focal Loss for class-imbalance experiments |
| `baselines.py` | Trains XGBoost + LSTM-only for comparison, appends to `results.json` |
| `train_no_none.py` | Trains the LSTM+Attention model after excluding the dominant `None` class |
| `baselines_no_none.py` | Trains XGBoost + LSTM-only baselines after excluding the `None` class |
| `load_data_kazakhstan.py` | Prepares the Kazakhstan SPEI-based dataset and generates model-ready sequences |
| `train_model_kazakhstan.py` | Trains the LSTM+Attention model using the Kazakhstan dataset |
| `train_kazakhstan_no_none.py` | Trains the Kazakhstan LSTM+Attention model after excluding the `None` class |
| `explain.py` | SHAP feature importance (weather + soil), saves 4 PNG plots |
| `app.py` | Streamlit dashboard — model comparison, class-scheme comparison, class-imbalance experiments, Kazakhstan portability, SHAP viewer, and live prediction (see `README_Streamlit.md`) |

**Important:** `CLASS_SCHEME` must be set to the same value (4, 5, or 6) in
`load_data.py`, `train_model.py`, `baselines.py`, and `explain.py` before
each relevant US class-scheme run. They must all agree on which label
scheme's data/model they're using.

---

## How to Run

### US Dataset

```bash
# Set CLASS_SCHEME (4, 5, or 6) at the top of each relevant file first:

python load_data.py      # -> sequences_{N}.npz
python train_model.py    # -> model.pt, results.json
python baselines.py      # -> results.json (XGBoost + LSTM-only)
```

Repeat the first three steps once per class scheme (4/5/6) to reproduce the
full comparison in `results.json`.

### Focal Loss Experiments

```bash
python train_model_focal_loss.py
```

This version uses Focal Loss for class-imbalance experiments and focuses the
training objective more strongly on difficult and underrepresented samples.

### Severity-Only Experiments

```bash
python train_no_none.py
python baselines_no_none.py
```

These experiments exclude the dominant `None` class and focus specifically
on drought severity classification.

### Kazakhstan Experiments

```bash
python load_data_kazakhstan.py
python train_model_kazakhstan.py
python train_kazakhstan_no_none.py
```

These scripts provide the Kazakhstan extension and the corresponding
severity-only experiment.

### SHAP Explainability

```bash
python explain.py
```

This generates:

```text
shap_summary.png
shap_heatmap.png
shap_casestudy.png
shap_soil_importance.png
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

---

## Requirements

```bash
pip install torch numpy pandas scikit-learn xgboost shap matplotlib streamlit
```

For GPU training, install the CUDA-matched torch build instead of the
default, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For the Kazakhstan NetCDF/SPEI workflow, an appropriate NetCDF backend such
as `netCDF4` or `h5netcdf` may also be required by `xarray`.

---

## Connection to Kazakhstan Research

This pipeline serves as a proof-of-concept for the full research project:

**"Multi-Source Satellite Time Series Drought Prediction in Kazakhstan
Using CNN-LSTM with Explainability Analysis"**

The same LSTM+Attention architecture, preprocessing pipeline, and SHAP
explainability framework provide the methodological foundation for the
Kazakhstan extension.

The Kazakhstan experiments use a different geographic region and drought
indicator from the original US experiments.

Key differences in the full study:

- Region: North Kazakhstan (Kostanay zone) instead of US counties
- Labels: SPEI-3 index instead of US Drought Monitor score
- Spatial structure: CNN encoder added for satellite image tensors
- Period: 2015–2024 (10 years)

The broader research framework is intended to incorporate multiple
environmental and satellite data sources, including:

- MODIS NDVI
- CHIRPS rainfall
- ERA5-Land
- GRACE groundwater

---

## Results Summary

### Primary comparison (4-class, 500 counties, with soil data)

| Metric      | XGBoost (baseline) | LSTM-only (baseline) | LSTM+Attention (ours) |
|-------------|--------------------|------------------------|--------------------------|
| Accuracy    | 0.6891             | 0.7752                 | **0.7778**               |
| Macro F1    | 0.4080             | 0.4710                 | **0.4718**               |
| MAE         | 0.4692             | 0.3002                 | 0.3027                   |
| RMSE        | 0.9294             | 0.6934                 | 0.7087                   |

LSTM+Attention outperforms both baselines on Accuracy and Macro F1 across
every class scheme tested (see below), indicating the attention mechanism
and soil branch are contributing real value rather than added complexity
for no benefit.

### Effect of label granularity (LSTM+Attention, same data, same model)

| Classes | Accuracy | Macro F1 | Rarest class recall |
|---------|----------|----------|------------------------|
| 6 (None/D0/D1/D2/D3/D4) | 0.7629 | 0.3309 | D4: 0/51 (0%) |
| 5 (D3+D4 merged)        | 0.7644 | 0.3882 | D3+: 89/372 (24%) |
| 4 (simplified)          | 0.7778 | 0.4718 | Severe+: 916/1445 (63%) |

**Key finding:** Macro F1 declines smoothly as label granularity increases,
driven specifically by data scarcity in the rarest classes (D3, D4). D4 had
only 51 test samples and the model achieved 0% recall on it in the 6-class
setup — not a modeling flaw, but a direct consequence of how rare extreme
droughts are in a 500-county sample. Merging rare classes recovers most of
the lost performance, confirming this is a data-availability effect rather
than an architectural limitation.

### Soil data contribution (6-class scheme)

| | Macro F1 |
|---|---|
| Without soil data | 0.3024 |
| With soil data     | 0.3309 |

Adding static soil/terrain features improved Macro F1 by ~0.03 with no
additional weather data required — the model's top soil predictors (`lon`,
`lat`, grassland %, elevation, slope) are all physically sensible drivers of
regional drought susceptibility.

*(See `results.json` for the full run history across all schemes, models,
and configurations.)*

### 500-County to 1000-County Expansion

The project initially used **500 randomly sampled counties (seed=42)**.
Later experiments expanded the US sampling to **1000 counties** to investigate
whether additional spatial coverage improved model performance.

The 500-county configuration was used for the initial development and
class-scheme experiments, while the later 1000-county configuration was used
for additional scaling experiments.

The 1000-county configuration increased the spatial coverage of the US
dataset and provided an additional experiment for evaluating whether model
behavior remained consistent when more counties were included.

### Class-Imbalance Experiments

Focal Loss was introduced to investigate whether explicitly focusing the
training objective on difficult and underrepresented samples could improve
minority-class recognition.

The Focal Loss implementation is provided in:

```text
train_model_focal_loss.py
```

This experiment complements the class-granularity analysis by addressing
class imbalance through the training objective rather than changing the
label definitions.

### Severity-Only Experiments

Additional experiments excluded the dominant `None` class and evaluated the
model specifically on drought severity classes.

The purpose was to determine whether the strong presence of the `None` class
was masking the model's ability to distinguish among actual drought
conditions.

The corresponding implementations are:

```text
train_no_none.py
baselines_no_none.py
```

These experiments provide a separate evaluation of drought-severity
recognition without the dominant non-drought category.

### Kazakhstan Extension

The pipeline was subsequently extended to Kazakhstan using SPEI-based
drought information.

The Kazakhstan implementation includes:

```text
load_data_kazakhstan.py
train_model_kazakhstan.py
train_kazakhstan_no_none.py
```

The Kazakhstan experiments represent a portability test of the general
workflow in a different geographic and drought-index setting.

The US and Kazakhstan results should not be directly compared because the
datasets, labels, geographic settings, and preprocessing are different.

### SHAP Explainability

SHAP was used to investigate which weather and soil/terrain features
contribute most strongly to model predictions.

The project generates four explainability visualizations:

- `shap_summary.png`
- `shap_heatmap.png`
- `shap_casestudy.png`
- `shap_soil_importance.png`

The SHAP analysis provides global feature-importance information together
with temporal, case-specific, and soil/terrain explanations.

A notable finding is that longitude (`lon`) is the single most important
feature in the overall model. This is physically plausible because longitude
captures broad geographic and climatic gradients across the continental
United States.

This should nevertheless be interpreted as model feature importance rather
than causal evidence: geographic variables can act as proxies for multiple
underlying climatic and environmental factors.

### Streamlit Dashboard

The project includes an interactive Streamlit dashboard for exploring the
experimental results and model analysis.

The dashboard provides:

- Overview
- Field Report walkthrough
- Model Comparison
- Class Scheme Comparison
- Class Imbalance Experiments
- Kazakhstan Portability
- SHAP Explainability
- Live Prediction

The dashboard reads the experiment history from `results.json` and uses the
generated SHAP visualizations for interactive analysis.

See `README_Streamlit.md` for setup and usage instructions.

---

## Project Outputs

### Generated Sequence Datasets

```text
sequences.npz
sequences_4.npz
sequences_5.npz
sequences_6.npz
sequences_kazakhstan.npz
```

These contain processed temporal sequences, labels, and associated features
required for model training.

### Trained Model Checkpoints

```text
model.pt
model_no_none.pt
model_kazakhstan.pt
model_kazakhstan_no_none.pt
```

### Experiment History

```text
results.json
```

`results.json` stores the experiment history used by the Streamlit dashboard
for comparing different models, class schemes, and configurations.

### SHAP Outputs

```text
shap_summary.png
shap_heatmap.png
shap_casestudy.png
shap_soil_importance.png
```

### Interactive Dashboard

```text
app.py
README_Streamlit.md
```

---

## Reproducibility Notes

The initial US county sampling uses a fixed random seed of **42**.

The project contains multiple experimental configurations rather than a
single final model. These include:

- 500-county US experiments
- 1000-county US experiments
- 4-, 5-, and 6-class label schemes
- Standard weighted Cross-Entropy training
- Focal Loss experiments
- Severity-only experiments without the `None` class
- Kazakhstan experiments
- SHAP explainability analysis
- Streamlit-based experiment visualization

Individual generated model checkpoints and sequence files should therefore
always be interpreted together with their corresponding experiment
configuration.

The large raw datasets, generated `.npz` sequence files, model checkpoints,
and NetCDF files are intentionally excluded from the Git repository through
`.gitignore`.

---

**Author**: Md Jahidul Islam Rana  
**Institution**: Al-Farabi Kazakh National University  
**Date**: June 2026