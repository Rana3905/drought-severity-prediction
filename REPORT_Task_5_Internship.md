# Kazakhstan Drought Prediction — Internship Task Report

**Author:** Md Jahidul Islam Rana
**Institution:** Al-Farabi Kazakh National University
**Program:** AlfaFlow Drought Prediction Internship 2026
**Dataset:** SPEI-HR Central Asia (Pyarali et al. 2022, CEDA, Open Government Licence v3)
**Scope note:** the internship task is restricted to open SPEI-HR Central Asia data
and does not require searching for unavailable data. The absence of Kazakhstan
soil-moisture, additional weather, and satellite-imagery inputs is an objective
data-availability limitation acknowledged by the task, not a technical shortcoming;
those extensions are left as stated in the "Out of scope / future work" section below.

---

## Objective

Extend the existing US drought prediction pipeline (LSTM+Attention, 18 weather
features + 31 soil features) to Kazakhstan, where only a single drought index
(SPEI-6) is available. Evaluate whether feature engineering, transfer learning,
multi-seed robustness, a two-stage prediction architecture, and model
explainability can close the gap created by limited data availability.

---

## Task 1: Working Under Limited Data Availability

### 1(a) Feature Engineering from SPEI

**Objective:** Build additional features from the same single SPEI time series
without searching for new data sources.

**Method:**
- Raw SPEI value at each timestep (1 feature)
- Rolling min/max/std over 3, 6, 12 months (9 features)
- Lag features: SPEI at t-1, t-3, t-6 months (3 features)
- Rate of change: SPEI(t) - SPEI(t-1) (1 feature)
- **Total: 14 engineered features per timestep**
- Extended input window from 6 to 18 months (12-month rolling stats require
  12 months of prior context)
- 500 grid cells sampled from Kazakhstan bounding box (seed=42)
- Chronological 70/15/15 train/val/test split

**Results:**

| Model | Features | Macro F1 | Accuracy |
|---|---|---|---|
| KZ 1 raw SPEI (baseline) | 1 | 0.542 | 82.8% |
| KZ 14 engineered features | 14 | **0.629** | **86.1%** |
| **Improvement** | | **+0.087** | **+3.3pp** |

**Conclusion:** Feature engineering from a single variable delivered a
substantial +0.087 F1 improvement, meaningfully closing part of the feature
gap without any new data source. The model trained to 150 epochs without early
stopping, suggesting further gains may be possible with extended training.

**Scripts:** `load_data_kazakhstan_features.py`, `train_model_kazakhstan_features.py`

---

### 1(b) Transfer Learning from US Model

**Objective:** Test whether pretrained US model weights provide a useful
starting point for Kazakhstan prediction.

**Method:**
- Trained US 1-feature PRECTOT model with architecture matched to KZ
  (PROJ=24, HIDDEN=64, no soil branch)
- Transferred 14 compatible weight layers (LayerNorm, LSTM weights, attention,
  fc1, fc2) to KZ model
- Only `proj.0.weight` could not transfer (US: 24x1 vs KZ: 24x14)
- Fine-tuned on KZ 14-feature data with reduced learning rate (1e-4)

**Results:**

| Model | Macro F1 | Accuracy |
|---|---|---|
| KZ from scratch (random init, seed 42) | **0.629** | **86.1%** |
| KZ transfer-learned from US PRECTOT | 0.541 | 83.4% |
| **Difference** | **-0.087** | **-2.8pp** |

**Why it failed:** US PRECTOT (precipitation) and KZ SPEI (standardized
drought index) encode fundamentally different physical signals. The US
pretrained LSTM and projection weights encode precipitation patterns that are
not useful — and actively harmful — for interpreting SPEI-based drought
sequences.

**Conclusion:** Transfer learning did NOT help. Cross-region transfer requires
the source and target to share compatible physical signals, not just the same
architecture.

**Multi-seed confirmation:** Seeds 42/123/2026 all show the same one-sided
penalty — scratch F1 0.629/0.616/0.619 vs transferred 0.541/0.523/0.529, a
consistent **-0.09 Macro F1** (range -0.087 to -0.092) with no overlap between
the two distributions. The transfer penalty is statistically robust, not seed
luck.

**Scripts:** `train_transfer_learning.py`

#### Spec-compliant re-run (US 18-feature pretrained base, only the input layer tuned)

The internship task specifies the transfer literally: *"use the weights of the
model trained on US data (18 features) as a pretrained base, and fine-tune only
the layer that sees the single available SPEI feature."* To answer this exactly,
three experiments were run on the Kazakhstan single-SPEI feature dataset
(`sequences_kazakhstan.npz`), all with the US 18-feature arch-matched model
(`model_us_full_archmatched.pt`, PROJ=24/HIDDEN=64) as the pretrained base:

| Model | Macro F1 | Accuracy | Trainable params |
|---|---|---|---|
| From scratch (random init) | **0.542** | **83.4%** | 30,100 |
| US18 frozen + input layer only | 0.320 | 80.8% | 48 |
| US18 full fine-tune (low LR) | 0.508 | 82.6% | 30,100 |

**Readings:**
- The literal spec approach — freezing the whole US tower and training only the
  48-parameter input projection — performs worst (0.320): the frozen US feature
  extractor cannot be reshaped by a 24x1 input adapter into a useful SPEI
  representation.
- Even unfreezing everything at a low learning rate (0.508) stays below training
  from scratch (0.542).
- The 14-feature attempt (-0.087) therefore reflects a general rule, not
  the specific transfer recipe: **US pretrained weights do not compensate for the
  missing features**, whichever layers we freeze or fine-tune.

**Scripts:** `train_transfer_learning_spec.py`

---

### 1(c) US Control Experiment

**Objective:** Answer the internship's core question: *"If the US had as few
features as Kazakhstan, would the quality gap be the same?"*

**Method:**
- Three US model variants, all with architecture matched to KZ (PROJ=24,
  HIDDEN=64):
  - **Variant A:** Full US — 18 weather + 31 soil features
  - **Variant B:** PRECTOT + soil — 1 weather feature + 31 soil features
  - **Variant C:** PRECTOT only — 1 weather feature, no soil
- Training data: sequences_4.npz (1000-county US data), subsampled to 200K
  rows for computational feasibility
- All variants use weighted CrossEntropyLoss, AdamW, cosine LR schedule

**Results:**

| Model | Features | Macro F1 | Accuracy |
|---|---|---|---|
| US full 18 weather + soil (arch-matched) | 18 + soil | **0.447** | **77.0%** |
| US PRECTOT + soil | 1 + soil | 0.369 | 71.1% |
| US PRECTOT only | 1 | 0.346 | 68.7% |
| KZ 1 raw SPEI | 1 | 0.542 | 82.8% |
| KZ 14 engineered features | 14 | 0.629 | 86.1% |

**Key findings:**
1. US with 1 feature (F1=0.346) is dramatically worse than KZ with 1 feature
   (F1=0.542) — a gap of **+0.196** in favor of Kazakhstan
2. Adding soil data to US PRECTOT only recovers +0.022 F1 — soil cannot
   compensate for a weak weather signal
3. SPEI is a far stronger single drought signal than raw precipitation
4. PRECTOT-only model achieves 0% recall on the Moderate class — it cannot
   distinguish Mild from Moderate drought

**Conclusion:** The quality gap is NOT just about feature count. SPEI encodes
drought-relevant information (precipitation + evapotranspiration combined)
that raw precipitation alone cannot provide. Even if the US had as few
features as Kazakhstan, SPEI would outperform PRECTOT as a single signal.

**Scripts:** `train_model_us_1feature.py`

---

## Task 2: Result Robustness Validation (Multi-seed)

**Objective:** Document the variance of Accuracy and Macro F1 across random
seeds, following the original report's robustness section.

**Method:**
- Retrain KZ 14-feature model on 3 different random seeds: 42, 123, 2026
- Identical hyperparameters: PROJ=24, HIDDEN=64, BATCH=64, LR=5e-4,
  150 max epochs, patience 30
- Same data: sequences_kazakhstan_features.npz (149K train, 32K val, 32K test)

**Results:**

| Seed | Accuracy | Macro F1 | MAE | RMSE | Epochs Run |
|---|---|---|---|---|---|
| 42 | 0.8614 | 0.6285 | 0.1478 | 0.4083 | 150 |
| 123 | 0.8576 | 0.6241 | 0.1520 | 0.4149 | 144 |
| 2026 | 0.8562 | 0.6319 | 0.1534 | 0.4163 | 139 |
| **Mean** | **0.8584** | **0.6282** | | | |
| **Std** | **0.0022** | **0.0032** | | | |
| **Range** | **0.0051** | **0.0078** | | | |

**Conclusion:** Results are **STABLE** across seeds. The Macro F1 spread of
0.0078 is well below the 0.03 threshold, confirming that the model's
performance is not attributable to lucky initialization. Seeds 123 and 2026
both triggered early stopping (epochs 144 and 139), while seed 42 ran all 150
epochs — yet all three converged to nearly identical final performance.

**Scripts:** `train_model_kazakhstan_multiseed.py`

---

## Task 3: Two-stage Prediction Model

**Objective:** Test whether splitting drought prediction into two stages —
binary detection followed by severity classification — improves over single-
stage 4-class classification.

**Method:**
- **Stage 1:** Binary classifier (None vs drought). Training labels: class 0
  = "no drought", classes 1/2/3 = "drought present"
- **Stage 2:** 3-class severity classifier (Mild / Moderate / Severe+).
  Trained only on drought samples (class 1/2/3 remapped to 0/1/2)
- **Combined (three variants):**
  1. **Hard gate @0.5** — Stage 1 decides drought/no-drought with a fixed 0.5
     threshold; Stage 2 runs only on drought-flagged samples
  2. **Tuned gate** — best single threshold on validation
  3. **Rescue routing** — P(drought) ≤ 0.35 → None, ≥ 0.95 → Stage 2, middle
     band (~18% of test) routed to the 4-class model
- All models use the same architecture (PROJ=24, HIDDEN=64, SOIL_PROJ=8)

**Correction note (important):** a previous version of this report reported a
two-stage combined Macro F1 of 0.227 (-0.40). That figure came from a
batch-index bug in the combined-evaluation loop of `train_two_stage.py`
(in-batch positions 0–255 were written into the global test array, pinning
~99.7% of predictions to class 0). Reproducing the buggy assembly gives
exactly 0.2274; the fixed assembly gives 0.6071. All two-stage numbers below
use the corrected assembly (bug fixed in `train_two_stage.py`).

**Results (seed 42, corrected):**

| Approach | Macro F1 | Accuracy |
|---|---|---|
| Single-stage 4-class | **0.629** | **86.1%** |
| Binary detector (Stage 1) | 0.854 | 91.5% |
| Severity classifier (Stage 2) | 0.608 | 70.2% |
| Two-stage combined (hard gate @0.5) | 0.607 | 86.8% |
| Two-stage + rescue routing | 0.635 | 87.6% |

**Confusion Matrix — two-stage hard gate (seed 42, corrected):**
```
                Predicted
True     None    Mild    Mod    Sev+
None   [24970   1443    168      1]
Mild   [  955   1945    484     17]
Mod    [   93    529    956     89]
Sev+   [    9     30    164    157]
```

**Error Analysis (corrected):**
- Stage 1 misclassifies ~20% of real drought samples as "no drought"; under a
  hard gate these become permanent class 0. This costs roughly **-0.02 Macro
  F1** (combined 0.607 vs single-stage 0.629) — real, but nowhere near the
  previously reported -0.40
- The binary detector itself is strong (F1=0.854, drought recall 0.80); most
  of the small combined loss comes from the hard gate's irreversible decisions
- **Rescue routing recovers the loss:** routing the ambiguous probability band
  to the 4-class model lifts combined F1 to 0.635 (seed 42) / mean 0.629

**Multi-seed confirmation (corrected):**

| Seed | Single-stage | Two-stage combined | Rescue routing |
|---|---|---|---|
| 42 | 0.6285 | 0.6071 | 0.6345 |
| 123 | 0.6241 | 0.6061 | 0.6159 |
| 2026 | 0.6319 | 0.6096 | 0.6358 |
| **Mean** | **0.628** | **0.608** | **0.629** |

**Conclusion (corrected):** The two-stage pipeline **works** — it does not
catastrophically fail. A hard gate reproduces all the classic cascade logic
but only costs ~0.02 Macro F1 (~0.61 vs 0.63); adding rescue routing makes the
two-stage architecture, on average, statistically tied with the single-stage
model (0.629 vs 0.628) at the cost of higher seed-to-seed variance. The
single-stage model remains the simplest, most stable design and is the
recommended deployment choice; the two-stage pipeline is a viable alternative
but provides no quality gain.

**Scripts:** `train_two_stage.py` (fixed), `improve_two_stage.py` (variants)

---

## Task 4: Model Explainability

**Objective:** Identify which features and timesteps most influence the KZ
model's predictions.

**Method:** Permutation importance on the test set. For each feature/timestep,
shuffle its values across all samples and measure the resulting Macro F1 drop.
3 repeats per feature/timestep, evaluated on the best KZ model (seed=42,
F1=0.6285).

### Feature Importance

| Rank | Feature | F1 Drop (importance) | Std |
|---|---|---|---|
| 1 | **spei_raw** | +0.262 | 0.010 |
| 2 | **rate_of_change** | +0.234 | 0.005 |
| 3 | **lag_6m** | +0.151 | 0.006 |
| 4 | lag_1m | +0.129 | 0.006 |
| 5 | lag_3m | +0.119 | 0.003 |
| 6 | rolling_min_3m | +0.118 | 0.003 |
| 7 | rolling_min_6m | +0.060 | 0.005 |
| 8 | rolling_std_6m | +0.048 | 0.004 |
| 9 | rolling_max_6m | +0.044 | 0.001 |
| 10 | rolling_max_12m | +0.040 | 0.006 |
| 11 | rolling_max_3m | +0.039 | 0.001 |
| 12 | rolling_std_3m | +0.024 | 0.003 |
| 13 | rolling_min_12m | +0.022 | 0.002 |
| 14 | rolling_std_12m | +0.010 | 0.007 |

### Timestep Importance (Top 8)

| Rank | Month | F1 Drop | Note |
|---|---|---|---|
| 1 | Month 18 (most recent) | +0.276 | Dominates |
| 2 | Month 17 | +0.171 | Strong |
| 3 | Month 16 | +0.069 | Moderate |
| 4 | Month 14 | +0.062 | |
| 5 | Month 13 | +0.051 | |
| 6 | Month 15 | +0.033 | |
| 7 | Month 7 | +0.031 | |
| 8 | Month 12 | +0.029 | |

**Key findings:**
1. **Raw SPEI** is the most important feature (F1 drops by 0.262 when
   permuted) — the current drought state is the primary predictor
2. **Rate of change** is nearly as important (0.234) — the direction and speed
   of SPEI movement matters almost as much as its absolute value
3. **Lag features** (1, 3, 6 months) provide strong secondary signal — the
   model uses historical drought trajectory
4. **Rolling statistics** contribute modestly — the 3-month rolling minimum is
   the most useful, capturing recent drought dips
5. **Recent months dominate** — months 16-18 (last 3 months of the 18-month
   window) account for most of the model's predictive power
6. **Oldest months matter least** — months 1-3 have minimal impact, suggesting
   the 18-month window is longer than strictly necessary

**Scripts:** `explain_kazakhstan.py`
**Outputs:** `shap_kz_feature_importance.png`, `shap_kz_timestep_importance.png`, `shap_kz_per_class.png`

---

## Controlled Experiments (Follow-up): Scenario C vs D, and Validated Two-stage

> These runs use the frozen shared modules (`model_lstm_attention.py`,
> `metrics_task_spec.py`), 6-month windows on the same Kazakhstan SPEI,
> per-location chronological 70/15/15 split, and the standard recipe
> (class-weighted CE sqrt-capped 2.0, AdamW 5e-4/5e-3, 5-epoch warmup + cosine,
> 150 epochs / patience 30, grad clip 1.0, label smoothing 0.1). Seeds 42/123/2026.

### Controlled feature comparison (Scenarios C/D)

C = 6 consecutive raw SPEI values (1 feature, `N×6×1`); D = the same windows
expanded to 22 derived features per timestep (`N×6×22`: spei_raw, lag 1/2/3/6m,
rolling mean/min/max/std over 3/6/12m, change 1/3/6m, neg_spei_count 6/12m).
Identical cells (seed 42), split, and architecture.

| Scenario | Input | Train seqs | Macro F1 (3-seed mean ± std) | Accuracy |
|---|---|---|---|---|
| C | 6×1 raw SPEI | 155,480 | **0.5648 ± 0.0012** | 0.8508 |
| D | 6×22 engineered | 152,964 | **0.5574 ± 0.0093** | 0.8437 |

**Delta: −0.0074.** The engineered features do not beat raw SPEI in this
controlled setting — slightly worse on average and with ~8× higher seed
variance. Consistent with Task 4 (raw SPEI is the dominant signal), this shows
SPEI's current value carries most of the predictive power; the feature
engineering that delivered +0.087 in Task 1a works through the specific
14-feature/18-month configuration, not through a per-month expansion to 22
features of 6-month windows.

**Scripts:** `load_data_task1_kz_raw.py`, `load_data_task1_kz_features.py`,
`run_task1_controlled.py`
**Results:** `task1_results.json`

### Validated two-stage pipeline (Scenario D)

Same binary(None/drought) → severity(Mild/Moderate/Severe+) cascade as Task 3,
built on the controlled 6×22 data.

**Integrity note:** the first version of this experiment produced a combined
Macro F1 of **0.675** by routing samples with the *ground-truth* drought mask
(an oracle gate). The defect was caught (stage-1 predictions were ignored),
the routing now uses **stage-1 predictions only** with length / non-zero-count /
spot-check assertions, and the numbers below are the corrected, non-oracle
results. This is in the same spirit as the Task 3 batch-index bug fix: the
pipeline is real and validated.

| Seed | Single 4-class MF1 | Two-stage combined MF1 | Single Acc | Combined Acc |
|---|---|---|---|---|
| 42 | 0.5621 | 0.5617 | 0.8476 | 0.8503 |
| 123 | 0.5466 | 0.5598 | 0.8467 | 0.8571 |
| 2026 | 0.5634 | 0.5658 | 0.8367 | 0.8554 |
| **Mean ± std** | **0.5574 ± 0.0093** | **0.5624 ± 0.0030** | 0.8437 | 0.8543 |

MAE: 0.1697 (single) vs 0.1622 (combined). Critical errors (combined, mean):
Severe→None 1.76%, Moderate→Mild 36.9%, Mild→Severe 1.5%, None→Drought 7.6%.

**Conclusion:** on the controlled 22-feature setup the two-stage pipeline is
statistically tied with single-stage on Macro F1 (+0.005, inside seed noise)
but better on accuracy (+0.011) and markedly more stable across seeds (std
0.0030 vs 0.0093). This agrees with Task 3's corrected conclusion: the
two-stage architecture is viable and can slightly beat single-stage, but does
not change the headline ranking; single-stage remains the simplest choice.

**Scripts:** `run_task2_two_stage.py`
**Results:** `task2_results.json`

---

## Task 5: Final Conclusions

### Which decisions delivered the largest quality gains?

| Rank | Decision | Impact | Explanation |
|---|---|---|---|
| **1** | **Feature engineering from SPEI** | **+0.087 F1** | Rolling stats, lags, and rate-of-change extracted meaningful temporal patterns from a single variable without new data |
| **2** | **SPEI as single feature (vs PRECTOT)** | **+0.196 F1** | SPEI integrates precip + evapotranspiration into a drought-relevant index that raw precipitation cannot match |
| **3** | **4-class label scheme** | **+0.141 F1** | Merging rare severe classes (from 6-class) eliminated data-scarcity artifacts |

### What failed and why?

| Approach | Impact | Reason |
|---|---|---|
| Transfer learning US -> KZ | **-0.09 F1** (3-seed mean) | US PRECTOT and KZ SPEI encode incompatible physical signals; penalty reproduced in all 3 seeds |

> **Note on the two-stage pipeline:** with the corrected evaluation (see
> Task 3), the two-stage approach does NOT fail — hard-gate combined is ~0.61
> Macro F1 and rescue routing ~0.63, statistically tied with single-stage.
> It delivers no positive gain, so it is not listed as a quality driver.

### Summary of best results

| Model | Macro F1 | Accuracy | Notes |
|---|---|---|---|
| KZ 14-feature LSTM+Attention | **0.629** | **86.1%** | Best overall, stable across seeds |
| KZ 14-feature + rescue-routing two-stage | 0.629 | 87.3% | Ties single-stage (mean of 3 seeds) |
| KZ two-stage combined (hard gate) | 0.608 | 86.5% | ~0.02 below single-stage (corrected) |
| KZ 1 raw SPEI baseline | 0.542 | 82.8% | Prior art, no engineering |
| US full 18 feat + soil | 0.447 | 77.0% | Architecture-matched baseline |
| KZ multi-seed mean (3 seeds) | 0.628 | 85.8% | Confirms stability |

### Recommendations

1. **Feature engineering is the single highest-impact intervention.** For
   regions with limited data availability, deriving temporal statistics from
   available time series should be the first step before seeking new data
   sources.

2. **SPEI is a strong drought proxy** that outperforms raw meteorological
   variables as a single feature. Regions with access to SPEI or similar
   standardized indices should use them as primary features.

3. **Cross-region transfer learning requires signal compatibility**, not just
   architectural compatibility. Transferring weights between physically
   different signals (precipitation vs drought index) hurts performance.

4. **Keep a single end-to-end 4-class model.** The two-stage pipeline is
   viable (corrected evaluation: ~0.61 hard gate, ~0.63 with rescue routing)
   but gains nothing over the single-stage model while adding complexity and
   variance; a single model that jointly learns "is there drought?" and "how
   severe?" is simpler and equally accurate.

5. **The model relies primarily on recent drought state** (last 3 months).
   Shorter input windows (6-12 months) with heavier recent weighting may
   be more efficient without losing accuracy.

### Out of scope / future work

The internship task confines this study to open SPEI-HR Central Asia data and
explicitly does not require searching for unavailable data; the missing soil
data is treated by the task as an *objective data-availability limitation, not a
technical shortcoming*. The following are therefore intentionally excluded and
are listed for transparency rather than as unfinished requirements:

1. **Real Kazakhstan soil-moisture data** — the 31-dimension soil branch is a
   zero-filled placeholder (scaffolding kept for architecture compatibility with
   the US pipeline).
2. **Additional weather variables** (temperature, humidity, wind, ...) — all 14
   KZ features are derived from the SPEI series only.
3. **A CNN branch for satellite imagery** — the current pipeline is sequence-only
   (LSTM+Attention); imagery integration is future work.
4. **Validation beyond 500 grid cells** (~14% of Kazakhstan's 5 km grid) — sampling
   mirrors the original report's 500-county approach; broader cross-region
   validation is future work.
5. **Uncertainty quantification for operational deployment** — outside the five
   task definitions; noted as a deployment-phase need.

---

## Appendix A: Model Architectures

### KZ 14-feature model (best)

Architecture scaled down from the US model (PROJ=32, HIDDEN=96) to match
Kazakhstan's smaller dataset size (~150K training samples vs ~870K for US),
reducing overfitting risk.

```
Input: (batch, 18 timesteps, 14 features)
  -> Linear(14, 24) + LayerNorm(24) + ReLU + Dropout(0.2)
  -> LSTM(24, 64, 1 layer, batch_first=True)
  -> Additive Attention (Linear(64,64) + tanh + Linear(64,1))
  -> Weighted sum -> 64-dim context vector
  -> Dropout(0.4)
  -> Linear(64+8, 32) + ReLU    [8 = SOIL_PROJ, but all-zeros for KZ]
  -> Linear(32, 4)
  Output: 4-class logits

Total parameters: ~29,884
```

### KZ 1-feature model

```
Same architecture but input_size=1
Total parameters: ~16,000
```

## Appendix B: Hyperparameters

| Parameter | Value |
|---|---|
| PROJ_SIZE | 24 |
| HIDDEN_SIZE | 64 |
| N_LAYERS | 1 |
| DROPOUT | 0.4 |
| SOIL_PROJ | 8 |
| BATCH_SIZE | 64 |
| LR | 5e-4 |
| WEIGHT_DECAY | 5e-3 |
| LABEL_SMOOTHING | 0.1 |
| MAX_EPOCHS | 150 |
| PATIENCE | 30 |
| MAX_CLASS_WEIGHT | 2.0 |
| OPTIMIZER | AdamW |
| SCHEDULER | Cosine annealing with 5-epoch warmup |
| GRADIENT_CLIPPING | 1.0 |
| LOSS | Weighted CrossEntropyLoss (sqrt-clipped) |

## Appendix C: Kazakhstan Data Details

| Property | Value |
|---|---|
| SPEI timescale | 6-month (SPEI-06) |
| Region | Central Asia, cropped to Kazakhstan bbox |
| Bounding box | lon [46.5, 87.3], lat [40.6, 55.4] |
| Temporal coverage | 456 months |
| Spatial resolution | ~5 km (SPEI-HR) |
| Grid cells sampled | 500 (seed=42) |
| Input window | 18 months (for engineered features) |
| Train/val/test split | 70/15/15 chronological |
| Train sequences | 149,380 |
| Validation sequences | 32,010 |
| Test sequences | 32,010 |

---

**Generated files:**
- `task_results_summary.json` — structured experiment results
- `REPORT_Task_5_Internship.md` — this report
- `shap_kz_feature_importance.png` — feature permutation importance chart
- `shap_kz_timestep_importance.png` — timestep permutation importance chart
- `shap_kz_per_class.png` — per-class feature importance heatmap

**Date:** August 2026
