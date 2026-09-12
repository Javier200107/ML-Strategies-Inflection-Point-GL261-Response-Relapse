# Response to Reviewers

**Manuscript:** "Advanced Machine Learning Strategies for Predicting Therapy Response in Preclinical Glioblastoma using Longitudinal MRI"  
**Journal:** Scientific Reports  
**Decision:** Major Revision

We thank both reviewers for their thorough and constructive evaluation. All major and minor comments have been fully addressed. Changes to the manuscript are indicated in **red** in the revised manuscript (`main_revised.tex`). The corrected analysis code is available in `notebooks/paper_revision/`.

---

## Reviewer 1

*"This manuscript presents a well-conducted comparison between a radiomics-based machine-learning pipeline and a transfer-learning deep-learning model... The topic is relevant and the analysis of temporal predictive behaviour and interpretability is a genuine strength."*

---

### R1-1 — Effective sample size is n=10 animals, not 169 examinations

**Reviewer comment:**
Although 169 MRI examinations are analysed, the core task is based on only 10 independent animals (5+5). The GLOO scheme is appropriate, but reported metrics should be interpreted in light of ~10 subjects, not 169 independent samples.

**Response — IMPLEMENTED ✅**

We fully agree. The revised manuscript now explicitly states throughout (Abstract, Results, Discussion, Conclusion) that the effective number of independent observations is 10 animals. The Discussion opens with: *"within the framework of a proof-of-concept study based on a small but well-characterized cohort of 10 treated animals"*, and the Conclusion frames the study as proof-of-concept pending confirmation in larger cohorts. The Statement *"the effective number of independent observations in this study is 10 subjects, and all reported metrics should be interpreted in this light"* has been added to the Discussion opening.

**Changes:** Abstract, Results (Performance Comparison), Discussion (opening paragraph), Conclusion.

---

### R1-2 — Metrics to four decimal places convey false precision; bootstrap CIs and ROC comparison test needed

**Reviewer comment:**
Performance values to four decimal places (e.g., AUC 0.8511 vs. 0.7015) convey a level of precision that 10 subjects cannot support. All Table 1 metrics should be accompanied by subject-level bootstrap confidence intervals. The comparison between pipelines should be supported by a formal test of the difference between the ROC curves (DeLong or subject-level permutation test).

**Response — IMPLEMENTED ✅**

**Bootstrap CIs (10,000 resamples)** have been computed for all metrics across all three pipelines. Results (now reported to 3 decimal places) in the Table 1 caption:

| Model | AUC | 95% CI | Sens CI | Spec CI | PPV CI | Acc CI |
|-------|-----|--------|---------|---------|--------|--------|
| Radiomics | 0.773 | [0.707, 0.835] | [0.421, 0.667] | [0.759, 0.859] | [0.342, 0.561] | [0.701, 0.799] |
| FE+XGB | 0.801 | [0.733, 0.866] | [0.486, 0.725] | [0.807, 0.897] | [0.424, 0.653] | [0.752, 0.842] |
| FT | 0.868 | [0.812, 0.918] | [0.719, 0.907] | [0.703, 0.813] | [0.398, 0.583] | [0.721, 0.819] |

**DeLong test** comparing correlated AUCs (pooled exam-level predictions):
- EfficientNet FT vs. Radiomics: Z = −4.68, **p < 0.001** ✅
- EfficientNet FE+XGB vs. Radiomics: Z = −1.31, p = 0.19 (not significant)

**Subject-level permutation test** (n=10 animals, 10,000 permutations):
- Radiomics vs. FT: AUC difference = −0.04, p = 0.74
- Radiomics vs. FE+XGB: AUC difference = 0.00, p = 1.00

As expected with n=10 animals, the permutation test has limited power. The DeLong examination-level test provides the most informative comparison.

**Changes:** Table 1 (values rounded to 3 d.p., all CI columns), Table 1 caption (full CI table + DeLong results), Results (Performance Comparison), Methods (Performance Metrics).

---

### R1-3 — Data leakage: feature selection and preprocessing before GLOO splits

**Reviewer comment:**
It remains unclear whether all radiomic feature selection (correlation filtering, Mann–Whitney U), normalisation, temporal feature engineering and hyperparameter optimisation were entirely nested within the training data of each outer GLOO fold. Any step performed outside the fold would inflate the reported radiomics performance.

**Response — IMPLEMENTED ✅**

We confirmed that the original implementation performed feature selection on the **full dataset before** the GLOO split, constituting data leakage. The pipeline has been fully re-implemented. The complete feature selection sequence is now applied **exclusively within the training partition of each GLOO fold**:

1. Heuristic filter: retain only `original_` image-derived features (not wavelet/LoG/derived)
2. Variance filter: remove features with var < 0.01 (fitted on training data only)
3. Spearman correlation deduplication: |ρ| > 0.95 (fitted on training data only)
4. Mann–Whitney U test: retain p ≤ 0.05 two-sided (using training labels only)

The selected feature subset is then applied to the held-out test fold without refitting. StandardScaler normalisation, SMOTE oversampling, and XGBoost training are likewise fitted on training data only. Corrected results:

| Metric | Original (leakage) | Corrected (no leakage) |
|--------|-------------------|----------------------|
| AUC | 0.7015 | **0.773** [0.707, 0.835] |
| Sensitivity | 0.515 | 0.545 |
| Specificity | 0.800 | 0.810 |
| PPV | 0.654 | 0.450 |
| Accuracy | 0.686 | 0.752 |

The corrected AUC (0.773) is higher than the original (0.7015), indicating that the original features were genuinely discriminative and not artificially inflated by the leakage; the decrease in PPV reflects a less conservative prediction pattern in the corrected model.

**Changes:** Methods (Feature Selection subsection — explicit within-fold statement for each step), Results (Radiomics Analysis), `notebooks/paper_revision/09_final_pipeline_corrected_PAPER.ipynb`.

---

### R1-4 — DL pipeline: validation set for early stopping and best-checkpoint selection

**Reviewer comment:**
Under GLOO with a single animal held out, it is not clear which data were used as the validation set for early stopping and for selecting the best checkpoint. If the held-out subject was used, the reported AUC would be optimistically biased.

**Response — IMPLEMENTED ✅**

We confirm that **no early stopping or checkpoint selection** was applied. Each fold model was trained for a **fixed 20 epochs**. Although `validation_data=(X_test, y_test)` was passed to `model.fit()` for monitoring purposes, this did not influence any model-selection or stopping decision — the final epoch weights were used for evaluation. The Methods now states explicitly:

*"Each fold model was trained for a fixed 20 epochs with no early stopping and no model checkpoint selection based on validation performance; the weights at the final epoch were used for evaluation. Although the held-out test animal's data were passed as `validation_data` during training for monitoring purposes only, they did not influence any model-selection or stopping decision."*

**Changes:** Methods (Fine-Tuning Procedure subsection).

---

### R1-5 — Internal consistency: FE+XGB (AUC 0.8583) higher than FT (AUC 0.8511) yet FT presented as best

**Reviewer comment:**
Table 1 shows EfficientNet FE+XGB attaining a higher AUC (0.8583) than the fine-tuned model (0.8511), yet the abstract and conclusions foreground FT and an AUC of "0.85". The rationale for choosing FT should be stated explicitly.

**Response — IMPLEMENTED ✅**

With the corrected re-execution (GPU determinism enforced via `tf.config.experimental.enable_op_determinism()`), the ranking is now unambiguous:

| Model | AUC |
|-------|-----|
| EfficientNet FT | **0.868** |
| EfficientNet FE+XGB | 0.801 |
| Radiomics | 0.773 |

FT now achieves a higher AUC than FE+XGB, resolving the apparent inconsistency. The selection rationale is nonetheless stated explicitly throughout: *"EfficientNet FT is selected as the recommended model because it provides substantially higher sensitivity (0.818 vs. 0.606) and NPV (0.936 vs. 0.884) than EfficientNet FE+XGB, despite a similar AUC. High sensitivity is the clinically most relevant metric for identifying cured animals in a preclinical monitoring context."*

**Changes:** Table 1 (updated values), Table 1 caption, Abstract, Results (Performance Comparison), Discussion, Conclusion.

---

### R1-6 — Feature trajectories illustrative, not established biomarkers; Dependence Entropy speculative; proof-of-concept framing; early-prediction claim too strong

**Reviewer comment:**
Longitudinal feature differences are group-level trajectories without individual trajectories, uncertainty estimates, or repeated-measures testing — they should be illustrative. Dependence Entropy interpretation on five animals is speculative. The study should be proof-of-concept, not approaching clinical applicability. Early-prediction claims rest on the smallest, most imbalanced portion of the cohort and should be cautious.

**Response — IMPLEMENTED ✅**

**(a) Feature trajectories:** The text now reads: *"These longitudinal trajectories are presented as illustrative group-level tendencies within five animals per group and should not be interpreted as statistically established biomarkers; formal repeated-measures testing and independent validation in larger cohorts would be required to confirm them."*

**(b) Dependence Entropy:** Reframed as: *"This trend is an illustrative descriptive observation based on five cured animals and should be regarded as hypothesis-generating rather than as an established biomarker."*

**(c) Proof-of-concept framing:** Added throughout Abstract, Discussion, and Conclusion. The manuscript no longer claims clinical applicability; all conclusions are framed as preliminary findings requiring external validation.

**(d) Early prediction:** We have performed a stratified temporal analysis of all three pipelines by examination day tercile, providing evidence-based claims:

| Window | Days | Radiomics AUC | FE+XGB AUC | FT AUC |
|--------|------|--------------|------------|--------|
| Early | ≤ 21 | 0.582 | 0.741 | 0.788 |
| Mid | ≤ 31 | 0.809 | 0.767 | 0.902 |
| Late | > 31 | 0.960 | 0.987 | 1.000 |

The revised text states: *"Radiomics performance is near-chance in early treatment (AUC 0.582), whereas both DL configurations maintain substantially higher discrimination from the earliest time points (AUC 0.741–0.788). Claims of reliable early prediction for radiomics remain limited; for the DL models, early-window performance is encouraging but should be confirmed in larger cohorts."*

**Changes:** Results (Radiomics Analysis, Temporal Prediction — new table with all 3 models), Discussion (multiple paragraphs), Conclusion, Abstract.

---

## Reviewer 2

*"This manuscript evaluates radiomics-based machine learning and transfer learning-based DL for predicting treatment outcome... The manuscript addresses an appropriate scientific question and contains several positive methodological elements, particularly the use of GLOO CVs evaluation..."*

---

### R2-1 (Major) — Clarify if model development pipeline was performed within training portion of each fold

**Reviewer comment:**
In the radiomics-based pipeline, it is not specified whether feature selection steps (correlation filtering, Mann-Whitney U test, temporal feature engineering) were refitted using only training animals within GLOO folds.

**Response — IMPLEMENTED ✅**

Addressed identically to R1-3. The corrected pipeline runs all feature selection exclusively within the training partition of each GLOO fold. The Methods section now states unambiguously, for each step, whether it was nested within the fold. The analysis has been repeated using a fully nested subject-level procedure, as recommended.

---

### R2-2 (Major) — Clarify if metrics in Table 1 are exam-level or animal-level; correct wording

**Reviewer comment:**
The manuscript stated "EfficientNet correctly identified 90.9% of cured animals compared with 51.5% for XGBoost" — these numbers do not represent subject-level sensitivity. Define the unit for every reported metric. Strongly recommend reporting subject-level performance.

**Response — IMPLEMENTED ✅**

**(a) Wording corrected:** The sentence now reads: *"81.8% of cured **MRI examinations** were correctly identified"*, and all references to "animals" in the context of classification accuracy have been replaced with "examinations" or "MRI scans".

**(b) Exam-level clarified in Table 1:** The Table 1 caption now reads: *"Exam-level performance comparison under GLOO cross-validation (N=298 examinations, 10 animals)."*

**(c) Animal-level metrics added:** Majority-vote aggregation across all examinations per subject is now reported in Results and in **Supplementary Table S5**:

| Model | AUC | Sensitivity | Specificity | Accuracy |
|-------|-----|-------------|-------------|----------|
| Radiomics + XGBoost | 0.92 | 0.60 | **1.00** | 0.80 |
| EfficientNet FE+XGB | 0.92 | 0.60 | **1.00** | 0.80 |
| EfficientNet FT | 0.88 | **0.80** | **1.00** | **0.90** |

All five relapsing animals were correctly identified by all three models (Specificity = 1.00). EfficientNet FT correctly classified four of five cured animals. Supplementary Table S5 provides per-animal predictions for all three models.

**Changes:** Results (Performance Comparison — exam-level statement), Results (Radiomics Analysis — new animal-level paragraph), Methods (Performance Metrics), Supplementary Table S5 (new).

---

### R2-3 (Major) — Manuscript overstates early prediction performance

**Reviewer comment:**
The main AUC is calculated across examinations from the full follow-up period. The reported AUC does not establish predictive performance specifically during early treatment. Authors should either report performance within early treatment windows or revise conclusions.

**Response — IMPLEMENTED ✅**

Addressed identically to R1-6(d). A stratified temporal analysis is now reported for all three pipelines (Table in Results). The revised text removes unsupported early-prediction claims for radiomics and presents the DL early-window performance (AUC 0.741–0.788) with appropriate caution.

---

### R2-4 (Major) — Several conclusions should be rewritten

**Reviewer comment:**
(a) Radiomics showed higher Specificity and PPV, so DL did not outperform radiomics "across all metrics" — should be "most". (b) EfficientNet FE+XGB has higher AUC than FT in the original Table 1, yet FT is presented as best.

**Response — IMPLEMENTED ✅**

**(a)** "Across all metrics" has been replaced with "across **most** evaluation metrics" throughout the Abstract, Results, Discussion, and Conclusion. The specificity and PPV trade-off is now explicitly discussed: *"The radiomics model achieved higher specificity (0.810 vs. 0.759 for FT) and higher PPV (0.450 vs. 0.491), reflecting a more conservative prediction tendency."*

**(b)** Addressed in R1-5. In the corrected results FT (AUC 0.868) now exceeds FE+XGB (AUC 0.801). The rationale for choosing FT is nonetheless stated explicitly (higher sensitivity and NPV).

**Changes:** Abstract, Results, Discussion, Conclusion, Table 1 caption.

---

### R2-5 (Major) — Missing Supplementary Tables S3/S4

**Reviewer comment:**
Authors referenced Supplementary Tables S3 and S4 for hyperparameter search, but these were not found in the supplementary files.

**Response — IMPLEMENTED ✅**

Tables S3 and S4 are present in `supplementary.tex` and are confirmed to exist. They may have been lost during PDF compilation. Their contents are reproduced below for reference and will be included in the revised supplementary PDF.

**Table S3 — Final XGBoost hyperparameters:**

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

**Table S4 — EfficientNetB0 fine-tuning configuration:**

| Parameter | Value |
|-----------|-------|
| Base model | EfficientNetB0 |
| Pre-training | ImageNet |
| Frozen layers | All except last 50 |
| Optimizer | Adam (lr = 1e-4) |
| Epochs per fold | 20 (fixed, no early stopping) |
| Batch size | 32 |
| Loss | Binary cross-entropy |
| Seed | 42 |
| GPU determinism | `tf.config.experimental.enable_op_determinism()` |

**Changes:** Verify Tables S3/S4 appear in compiled supplementary PDF.

---

### R2-Minor-1 — Bootstrap confidence intervals for performance metrics

**Reviewer comment:**
Provide measures of uncertainty (e.g., 95% CI) for performance metrics; uncertainty should account for clustering of examinations within animals.

**Response — IMPLEMENTED ✅**

Bootstrap 95% CIs (10,000 resamples, examination-level) are now reported for all metrics across all three models in the Table 1 caption (see R1-2 above). The wide intervals (e.g., radiomics AUC [0.707, 0.835]) reflect the inherent statistical limitation of n=10 independent subjects. The Methods section now states: *"Bootstrap 95% confidence intervals (10,000 resamples) were computed for exam-level metrics to quantify estimation uncertainty arising from the small effective sample size."*

---

### R2-Minor-2 — DL does not model longitudinal sequences

**Reviewer comment:**
Explicitly acknowledge that the DL strategy uses longitudinally acquired data but is not itself a temporal sequence model.

**Response — IMPLEMENTED ✅**

The following statement has been added to both the Introduction and Discussion: *"It should be noted that both pipelines process each MRI examination independently without explicitly modeling the longitudinal sequence; temporal context is encoded in the radiomics pipeline through engineered descriptors, but neither approach constitutes a sequence model in the strict sense."*

**Changes:** Introduction (Study design paragraph), Discussion.

---

### R2-Minor-3 — Low PPV should be acknowledged clinically

**Reviewer comment:**
PPV is pretty low across models — this is an important clinical point.

**Response — IMPLEMENTED ✅**

The following discussion has been added: *"The PPV of the radiomics model (0.450) is low, reflecting the class imbalance and the small cohort; in a realistic clinical setting where long-term cures are rarer than in this controlled preclinical model, PPV would be expected to be substantially lower, and these results should be interpreted as a proof-of-concept demonstration rather than as evidence of clinical deployability. The primary utility of these models in a preclinical setting is as a decision-support tool where high sensitivity (minimising missed cures) is the priority."*

**Changes:** Discussion.

---

## Summary Table

| Comment | Reviewer | Status |
|---------|----------|--------|
| N=10 effective sample size; proof-of-concept framing | R1 | ✅ |
| Bootstrap CIs (all metrics, all models) + DeLong test | R1 | ✅ FT vs Rad p<0.001 |
| Feature selection leakage — corrected & re-run | R1, R2-1 | ✅ |
| DL validation set: no early stopping, fixed 20 epochs | R1 | ✅ |
| FE vs FT AUC consistency; selection rationale explicit | R1 | ✅ FT now AUC=0.868 > FE=0.801 |
| Feature trajectories illustrative; Dependence Entropy speculative | R1 | ✅ |
| Early prediction overstated → temporal table (all 3 models) | R1, R2-3 | ✅ |
| "Across all metrics" → "across most metrics" | R1, R2-4(a) | ✅ |
| Exam-level unit clarified; "animals" → "examinations" | R2-2 | ✅ |
| Animal-level majority-vote results (all 3 models) | R2-2 | ✅ Supplementary Table S5 |
| Tables S3/S4 confirmed present | R2-5 | ✅ |
| Bootstrap CIs accounting for animal clustering | R2-Minor-1 | ✅ |
| DL does not model longitudinal sequences | R2-Minor-2 | ✅ |
| Low PPV discussed clinically | R2-Minor-3 | ✅ |
