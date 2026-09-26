## **Background** 

In the previous stage, XGBoost, LSTM-only, and LSTM + Attention models were evaluated for drought severity classification. The main benchmark was built on US data, where a richer set of climate, soil, and terrain features was available. 

The Kazakhstan experiment was an initial portability test. It used SPEI-HR Central Asia data, 500 sampled grid cells, 6-month rolling windows, and one input feature: SPEI. The results confirmed that the approach can be applied to Kazakhstan, but they do not yet separate the effect of limited feature availability from regional differences. 

This assignment consists of two independent but related tasks. 

# **Task 1** 

## **Controlled Analysis of Data-Limitation Effects** 

## **Objective** 

Determine how much model performance in Kazakhstan is limited by the lack of input features and how much is explained by differences between Kazakhstan and the United States. 

## **Main Research Question** 

How much does model performance decline when moving from the full US feature set to SPEI only, and can SPEI-derived features reduce this gap for Kazakhstan? 

## **Experimental Design** 

|Scenario|Dataset|Input Features|Purpose|
|---|---|---|---|
|A|United States|Full original feature set|Reference benchmark|
|B|United States|SPEI only|Measures the effect of feature limitation|
|C|Kazakhstan|Raw SPEI only|Current Kazakhstan baseline|
|D|Kazakhstan|SPEI + derived features|Tests improvement without new external data|



## **Required Work** 

## **1. Build comparable experiments** 

Use the same setup for all scenarios where possible: 

- Target classes: <mark>None, Mild, Moderate, Severe+</mark> 

- Time-series input window: 6 months 

- Comparable train, validation, and test split logic 

- At least three random seeds: 42, <mark>123,</mark> and <mark>2026</mark> 

## **2. Engineer features from SPEI** 

For the **Kazakhstan SPEI Enhanced** scenario, create features only from the available SPEI series: 

- SPEI lags for 1, 2, 3, and 6 months 

- Rolling mean over 3, 6, and 12 months 

- Rolling minimum and maximum over 3, 6, and 12 months 

- Rolling standard deviation over 3, 6, and 12 months 

- SPEI change over 1, 3, and 6 months 

- Number of negative-SPEI months during the previous 6 and 12 months 

## **3. Compare results** 

For every scenario and random seed, report: 

- Accuracy 

- Macro F1 

- MAE 

- RMSE 

- Precision by class 

- Recall by class 

- Special attention to Moderate and Severe+ Recall 

## **Deliverables** 

- One table containing results for all scenarios and random seeds 

- Mean, standard deviation, minimum, and maximum for key metrics 

- A comparison chart for Accuracy and Macro F1 

- A chart focused on Moderate and Severe+ Recall 

- A concise conclusion answering the main research question 

- Reproducible code and a README 

## **Definition of Done** 

The task is complete when you can state, with evidence: 

“The performance difference between the United States and Kazakhstan is primarily explained by feature limitations / regional differences / a combination of both.” 

The conclusion must be based on the four controlled scenarios, not assumptions. 

# **Task 2** 

## **Two-Stage Drought Detection and Severity Classification** 

## **Objective** 

Improve practical drought detection by separating two different questions: 

1. Is drought present? 

2. If drought is present, how severe is it? 

The previous work showed that the dominant <mark>None</mark> class can inflate Accuracy. It also showed that removing None improves Macro F1 but changes the problem: the model then classifies severity only among cases already known to be drought events. 

## **Main Research Question** 

Does a two-stage architecture improve the detection of real drought events, especially Moderate and Severe+ cases, compared with the current single-stage four-class model? 

## **Model Logic** 

Input: 6-month SPEI time series │ ▼ 

Stage 1 — Drought Detector 

Is drought present? 

│ ┌─────────┴─────────┐ ▼                                                    ▼ No drought                            Drought present → None                                              │ ▼ Stage 2 — Severity Classifier Mild / Moderate / Severe+ 

## **Required Work** 

## **1. Implement Stage 1 — Drought Detector** 

Train a binary classifier: 

|Output Class|Meaning|
|---|---|
|None|No drought|
|Drought|Mild, Moderate, or Severe+ drought|



### Report: 

- Accuracy 

- Precision 

- Recall 

- F1 score 

- Confusion matrix 

- Drought Recall, since missed drought events are operationally risky 

## **2. Implement Stage 2 — Severity Classifier** 

Train a separate classifier for drought cases only: 

|Output Class|Meaning|
|---|---|
|Mild|Low drought severity|
|Moderate|Medium drought severity|
|Severe+|Severe or extreme drought|



### Report: 

- Macro F1 

- Precision by class 

- Recall by class 

- Confusion matrix 

- Dedicated evaluation of Moderate and Severe+ 

## **3. Build the final pipeline** 

Combine both stages so that the model still produces one of four final classes: 

- None 

- Mild 

- Moderate 

- Severe+ 

Compare the two-stage system with the current single-stage four-class classifier. 

## **Critical Error Analysis** 

Analyze these errors separately: 

|Error Type|Why It Matters|
|---|---|
|Severe+ predicted as None|A dangerous missed severe-drought event|
|Moderate predicted as Mild|Underestimation of risk|
|Mild predicted as Severe+|Overestimation / false alarm|
|None predicted as Drought|Unnecessary alert or investigation|



## **Deliverables** 

- Working code for the two-stage pipeline 

- Comparison table: single-stage vs. two-stage model 

- Confusion matrices for both approaches 

- Precision and Recall for all classes 

- Comparison of Severe+ false-negative rates 

- A concise recommendation on whether the two-stage model should become the next baseline 

## **Definition of Done** 

The task is complete when there is a data-supported answer to: 

- “Does separating drought detection from severity classification improve the identification of 

Moderate and Severe+ events?” 

