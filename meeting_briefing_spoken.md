# Meeting Briefing — Spoken Summary (~5 minutes)

## Opening (30 sec)

"Good [morning/afternoon]. I'm here to present my internship work on
drought prediction for Kazakhstan using deep learning. The project scope
was five tasks: feature engineering, transfer learning from US drought data,
a US control experiment, model robustness testing, and an explainability
analysis. I also built a full training pipeline from scratch using PyTorch
with an LSTM-plus-attention architecture."

---

## Task 1 — Feature Engineering (1 min)

"Task 1a was feature engineering. The original Kazakhstan pipeline used one
raw SPEI value as input — a single number per timestep. I engineered 14
derived features from that same SPEI signal: rolling statistics over 3, 6,
and 12-month windows, lagged values at 1, 3, and 6 months, and a rate of
change metric.

This raised the F1 score from 0.5419 to 0.6285 — an improvement of 0.087.
That's the largest single gain across all five tasks, and it came from
better representing the temporal dynamics already present in the data,
not from adding new data sources."

---

## Task 1b — Transfer Learning (1 min)

"Task 1b was transfer learning. The idea: train a model on US drought data
— which has 874,000 sequences — then fine-tune it on Kazakhstan's 149,000
sequences, hoping the US model learns general drought patterns that
transfer.

It didn't work. The transfer-learned model scored 0.5412 versus 0.6285
from training on Kazakhstan data alone — a drop of 0.087. The reason is
the input signals are fundamentally different: the US model was trained on
PRECTOT, which is raw precipitation, while Kazakhstan uses SPEI, which is
a drought index that combines precipitation with potential evapotranspiration.
Even though the architectures matched, the pretrained weights encoded
relationships specific to precipitation that were actively harmful when
applied to SPEI. We confirmed this across three random seeds: the penalty
was roughly the same in all of them, so it's a systematic incompatibility,
not bad luck."

---

## Task 1c — US Control Experiment (45 sec)

"Task 1c was a US control experiment — the comparison that validates the
whole analysis. I trained three US models with the same architecture as the
Kazakhstan model: US full 18-feature, US 1-feature with soil, and US
1-feature with just PRECTOT. The US PRECTOT-only model scored 0.346. The
Kazakhstan SPEI-only model scored 0.542. That's a gap of +0.196.

This is the single most important finding: the quality difference is about
signal strength, not feature count. SPEI is simply a far stronger
predictor of drought than raw precipitation. One number — SPEI — beats
eighteen US features combined."

---

## Task 2 — Multi-seed Robustness (30 sec)

"Task 2 tested model stability across three random seeds: 42, 123, and
2026. The F1 scores were 0.6285, 0.6241, and 0.6319 — a spread of just
0.008. The model is stable and not sensitive to random initialization.
The reported performance is reliable."

---

## Task 3 — Two-stage Pipeline (30 sec)

"Task 3 tested a two-stage pipeline: a binary classifier to detect drought
first, then a severity classifier to grade severity only on drought
samples. An earlier analysis suggested it collapsed to 0.227 — but that
turned out to be a bug in the evaluation script, not the pipeline. After
fixing it, the corrected numbers show the two-stage pipeline scores about
0.61 Macro F1, which is just two points below the single-stage model's
0.63. I also tried an improved three-way routing variant — confident
cases go to their specialist stages, ambiguous ones go to the 4-class
model — and that ties single-stage at 0.63 across three seeds. So the
honest bottom line: the two-stage pipeline works, it just doesn't beat a
single end-to-end model, so single-stage remains the recommended, simplest
choice. The evaluation bug is fixed and documented."

---

## Task 4 — Explainability (30 sec)

"Task 4 used permutation importance to explain model behavior. The top
three findings: first, raw SPEI dominates at +0.262 — the model relies
heavily on the current drought index value. Second, rate of change is the
second most important feature at +0.234 — the model is tracking drought
trajectory, not just current state. Third, the 16-18 month window — the
most recent timesteps — account for 39% of total importance. The model
weights recent conditions most heavily, which aligns with meteorological
expectations."

---

## Controlled Follow-up Experiments (45 sec)

"Beyond the five tasks, I ran two controlled experiments using the frozen
shared model and metrics modules. First, I compared raw six-month SPEI
against the same windows expanded to 22 derived features — lags, rolling
statistics, and change rates. With identical cells, split, and architecture,
raw SPEI still won: 0.5648 against 0.5574 Macro F1. The per-month feature
expansion adds nothing on six-month windows — it reinforces that SPEI's
current value carries most of the signal.

Second, I rebuilt the two-stage pipeline on this controlled data, this time
strictly routing by what the binary gate predicts, not by the true labels.
One early version had accidentally used ground-truth routing, which inflated
the number to 0.675 — I caught that oracle-gate defect and fixed it before
reporting anything. The corrected two-stage ties the single model on Macro F1
at about 0.562, but improves accuracy by one point and is three times more
stable across seeds. So both experiments confirm the existing conclusions:
raw SPEI is the core signal, and two-stage is a viable but not decisive
alternative to a single end-to-end model."

---

## Closing — Key Takeaway (15 sec)

"The headline finding across all five tasks: a single SPEI value
outperforms the entire 18-feature US dataset by a wide margin — F1 0.542
versus 0.346, a gap of +0.196. The practical implication is that for
Kazakhstan drought prediction, the investment should go into better
drought index computation, not into collecting more raw variables.

One scope note: I didn't add soil, extra weather, or satellite imagery.
The internship task is restricted to open SPEI data and calls the missing
soil an acknowledged data-availability limitation, not a shortcoming — so
those are deliberately left as future work. Thank you — I'm happy to take
questions."
