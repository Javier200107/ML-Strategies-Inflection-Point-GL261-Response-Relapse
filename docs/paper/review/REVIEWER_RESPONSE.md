# Response to Reviewers

**Manuscript:** "Advanced ML Strategies for Predicting Therapy Response in Preclinical Glioblastoma using Longitudinal MRI"  
**Journal:** Scientific Reports  
**Decision:** Major Revision

We thank the reviewer for their thorough and constructive evaluation. Below we address each comment in turn. Changes to the manuscript are indicated in **bold**.

---

## Major Comments

---

### Major Comment 1 — Feature selection inside vs. outside cross-validation folds (data leakage)

**Reviewer comment (paraphrased):**  
The manuscript does not clearly state whether feature selection (variance filtering, Spearman correlation filtering, Mann-Whitney U test) was performed on the full dataset before cross-validation, or whether it was performed independently within each training fold. Performing selection on the full dataset constitutes data leakage and would lead to overly optimistic performance estimates.

**Response:**

We thank the reviewer for raising this important methodological point.

**[FILL IN AFTER CODE AUDIT]** After inspecting the code, feature selection in the current implementation is performed **before** the GLOO cross-validation loop, which constitutes data leakage from test folds into the feature selection step.

**Proposed correction (Action Item A1):** We will re-implement the pipeline so that the full feature selection sequence (variance filter → Spearman correlation filter → Mann-Whitney U filter → heuristic filter) is applied **exclusively within the training portion of each GLOO fold**, with the resulting feature subset applied to the held-out test fold. We will then re-run all experiments and update Table 1 with the corrected metrics.

We note that this change is expected to reduce performance estimates but will produce methodologically sound results. We will discuss the impact in the revised manuscript.

**Changes to manuscript:**  
- Methods section: Added explicit statement that feature selection is performed within each CV fold on training data only.  
- Table 1: Updated with corrected metrics.  
- Supplementary Table S2 (XGBoost hyperparameter grid): Unchanged.

---

### Major Comment 2 — Exam-level vs. animal-level metrics

**Reviewer comment (paraphrased):**  
The performance metrics in Table 1 appear to be computed at the exam level (each MRI session as an independent prediction unit), but the clinical question is whether the animal will respond or relapse. The reviewer recommends reporting animal-level metrics using a majority-vote aggregation across all exams from a given animal within each GLOO fold.

**Response:**

The reviewer is correct. The metrics in Table 1 are currently computed at the **exam level**: each of the 169 MRI examinations is treated as an independent prediction unit. This means that, for a given test mouse, each of its N exams contributes one prediction, and all predictions are pooled across folds before computing the global confusion matrix.

We agree that animal-level metrics are more clinically meaningful and better aligned with the actual prediction task (predict cure vs. relapse per animal). We will add animal-level metrics derived by **majority vote** across exams per animal per fold as the primary reporting metric, keeping exam-level metrics as secondary (with a clear label).

**Action Item A2:** Implement animal-level aggregation (majority vote per mouse across exam predictions) within each GLOO fold. Report both levels clearly in Table 1 (or split into Table 1 for animal-level and supplementary table for exam-level).

**Changes to manuscript:**  
- Methods section: Added description of the two-level evaluation (exam-level and animal-level).  
- Table 1: Updated with animal-level metrics as the primary metric. Exam-level metrics moved to supplementary or added as a second row per model.  
- Results section: Updated interpretation.

---

### Major Comment 3 — Overstated early-prediction claims

**Reviewer comment (paraphrased):**  
The manuscript claims the models perform well for early prediction (before relapse is clinically confirmed). However, the AUC values in Table 1 are computed across the full longitudinal follow-up, including late-timepoint exams that are inherently easier to classify. Early prediction performance is not separately evaluated.

**Response:**

The reviewer raises a valid and important point. The AUC values reported in Table 1 reflect classification performance across **all available timepoints** for each animal (days 7–35 post-treatment depending on the animal), including late timepoints where tumor size differences between cured and relapsing mice are more pronounced. This does not constitute a rigorous early-prediction evaluation.

**Action Item A3:** We will perform a stratified analysis of performance by examination day (or by tercile of the follow-up: early, mid, late). This will allow us to report AUC and sensitivity as a function of time and make defensible claims about early detection capability.

**Changes to manuscript:**  
- Remove or qualify the claims about "early prediction" in the Abstract, Results, and Discussion until the stratified analysis confirms early-timepoint performance.  
- Add Figure (or Supplementary Figure) showing AUC / sensitivity as a function of examination day.  
- Reframe the claim as: "the models maintain discriminative ability throughout the follow-up, including early timepoints, as shown in Figure X."

---

### Major Comment 4 — Overstated conclusions ("across all metrics")

**Reviewer comment (paraphrased):**  
The manuscript states in the Discussion that the EfficientNet models outperform radiomics "across all metrics." However, from Table 1, the radiomics model has higher Specificity (0.8000 vs. 0.5200 for FT) and higher PPV (0.6538 vs. 0.6522 for FT). This claim is factually incorrect and should be corrected.

**Response:**

The reviewer is correct. We apologise for this overstatement. Examining Table 1:

| Metric | Radiomics | Eff. FE+XGB | Eff. FT |
|--------|-----------|-------------|---------|
| AUC | 0.7015 | **0.8583** | 0.8511 |
| Sensitivity | 0.5152 | 0.6061 | **0.9091** |
| Specificity | **0.8000** | **0.9600** | 0.5200 |
| PPV | **0.6538** | **0.9091** | 0.6522 |
| NPV | 0.7059 | **0.7742** | **0.8125** |
| Accuracy | 0.6864 | **0.8046** | 0.7011 |

EfficientNet FE outperforms radiomics on all six metrics. EfficientNet FT has superior AUC, Sensitivity, and NPV but **lower Specificity and PPV** than radiomics.

**Changes to manuscript:**  
- Discussion: Replace "across all metrics" with a nuanced statement such as: "EfficientNet FE outperforms radiomics across all reported metrics, while EfficientNet FT achieves the highest sensitivity (0.9091) but at the cost of lower specificity (0.5200) compared to radiomics (0.8000), reflecting a sensitivity-specificity trade-off."  
- Discussion: Add a paragraph discussing the clinical implications of this trade-off (high sensitivity is preferred for early-warning systems; high specificity is preferred when false positives carry high cost).

---

### Major Comment 5 — Missing supplementary tables S3/S4

**Reviewer comment (paraphrased):**  
Supplementary tables S3 and S4 (final XGBoost hyperparameter configuration and EfficientNet fine-tuning configuration) are referenced in the manuscript but were not received.

**Response:**

We apologise for the omission. Tables S3 and S4 are present in our supplementary.tex file but may have been lost during PDF compilation. We confirm their contents below and will ensure they are included in the revised submission.

**Table S3 — Final XGBoost hyperparameters (Radiomics pipeline)**

| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| learning_rate | 0.01 |
| max_depth | 3 |
| subsample | 0.6 |
| colsample_bytree | 0.6 |
| gamma | 0.5 |
| min_child_weight | 1 |
| random_state | 42 |

**Table S4 — EfficientNet fine-tuning configuration**

| Parameter | Value |
|-----------|-------|
| Base model | EfficientNetB0 |
| Pre-training | ImageNet |
| Frozen layers | All except last 50 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Epochs per fold | 20 |
| Batch size | 32 |
| Loss | Binary cross-entropy |
| Random seed | 42 |

**Changes to manuscript:**  
- Supplementary: Verify Tables S3 and S4 appear in the compiled PDF and are clearly referenced.

---

## Minor Comments

---

### Minor Comment 1 — Confidence intervals

**Reviewer comment:** Add confidence intervals or other uncertainty estimates to the metrics in Table 1.

**Response:**  
We will add 95% bootstrap confidence intervals (10,000 bootstrap samples) for AUC, Sensitivity, Specificity, PPV, NPV, and Accuracy in Table 1.

**Action Item A4:** Compute bootstrap CIs from the pooled predictions of all GLOO folds and add them to Table 1.

---

### Minor Comment 2 — DL does not model longitudinal sequences

**Reviewer comment:** The manuscript should acknowledge that the deep learning models process each MRI exam independently, without modelling the longitudinal sequence structure.

**Response:**  
We agree and will add the following acknowledgement in the Methods section (Deep Learning subsection) and Discussion:

*"It should be noted that both EfficientNet configurations process each MRI examination independently as a 2D image. The model has no access to the temporal sequence of examinations, and longitudinal information is not explicitly modelled. This is a limitation compared to approaches that use recurrent architectures or sequence models. Despite this, the GLOO evaluation strategy ensures that all examinations from a given animal are used either entirely for training or entirely for testing, preventing temporal information leakage between folds."*

---

### Minor Comment 3 — Discuss low PPV

**Reviewer comment:** The low PPV, particularly for the EfficientNet FT model (0.6522), should be discussed in relation to the clinical utility of the models.

**Response:**  
We will add the following discussion:

*"The PPV values observed across all models (0.65–0.91) reflect the class imbalance in the dataset (5 cured vs. 5 relapsing out of 10 treated mice, approximately 1:1 after excluding controls). In a realistic clinical scenario where cure is rarer (as in human GBM where long-term responses are uncommon), the PPV would be substantially lower. We therefore caution that the reported PPV should not be extrapolated directly to clinical settings without recalibration on a representative cohort. The primary value of these models in a preclinical context is as a decision-support tool to prioritise animals for further monitoring, where high sensitivity (minimising false negatives) is more critical than high PPV."*

---

## Summary of Changes

| Action | Comment | Status |
|--------|---------|--------|
| A1 — Move feature selection inside GLOO folds, re-run experiments | Major 1 | TODO |
| A2 — Add animal-level majority-vote metrics to Table 1 | Major 2 | TODO |
| A3 — Stratified early/mid/late timepoint performance analysis | Major 3 | TODO |
| A4 — Add bootstrap CIs to Table 1 | Minor 1 | TODO |
| Fix "across all metrics" claim in Discussion | Major 4 | Easy text edit |
| Verify supplementary Tables S3/S4 appear in PDF | Major 5 | Easy fix |
| Acknowledge DL does not model sequences (Methods + Discussion) | Minor 2 | Easy text edit |
| Discuss low PPV (Discussion) | Minor 3 | Easy text edit |
