# Data Pipeline Brief — Student Risk Prediction

A foundation-level walkthrough of **what we feed in, what we compute, and what we output.** Read once; you won't need to reopen the notebook for the viva.

---

## 1. What's in the dataset

We use three CSV files from OULAD (Open University Learning Analytics Dataset):

| File | Rows | Key columns | What each row represents |
|---|---|---|---|
| `assessments.csv` | 206 | `id_assessment`, `code_module`, `code_presentation`, `assessment_type`, `date`, `weight` | **One assessment event** (a single CMA / TMA / Exam scheduled in a specific module & semester) |
| `studentAssessment.csv` | ~173,000 | `id_assessment`, `id_student`, `score` | **One student's submission** to one assessment (0–100 score) |
| `vle.csv` | 6,364 | `code_module`, `code_presentation`, `activity_type`, `week_from`, `week_to` | **One online learning resource** (video, quiz, forum, PDF…) available to students in a module-presentation |

**Mental model:** `assessments.csv` = list of exams. `studentAssessment.csv` = everyone's marksheet. `vle.csv` = catalog of study resources.

---

## 2. What we're predicting

For each assessment, a single **binary label**:

$$
\text{risk\_level} =
\begin{cases}
1 \; (\text{High Risk}) & \text{if at least 10\% of students failed this assessment} \\
0 \; (\text{Low Risk}) & \text{otherwise}
\end{cases}
$$

"Failed" means the student's `score` was below **40** (the OULAD pass mark).

**Why this label is useful:** a course coordinator planning next semester's presentation wants to know in advance which assessments are likely to be High Risk, so they can re-time the assessment, reinforce related VLE resources, or schedule extra support — *before* any student takes it.

---

## 3. `fail_rate` vs. `risk_level` — two different things

This trips people up. They are related but **not the same**.

| | `fail_rate` | `risk_level` |
|---|---|---|
| Type | Continuous (0.00 – 1.00) | Binary (0 or 1) |
| Example | 0.174 → 17.4% of students failed | 1 (because 17.4% ≥ 10%) |
| Role | **Intermediate** quantity — a measurement | **Final label** — what the model predicts |
| Used as a model feature? | **No** (never — would be target leakage) | **Never** (it's the target!) |
| Used for anything else? | Yes — used to compute `peer_fail_rate` for **other** assessments in the same module (leave-one-out, row's own value excluded) | Used only as the training target `y` |

**Flow:**

```
student scores  ──► fail_rate (per assessment)  ──► risk_level (threshold at 0.10)
                        │
                        └──► peer_fail_rate (leave-one-out, used as a feature)
```

So `fail_rate` appears twice in the pipeline: once to build the label, once (via the leave-one-out trick) to build a feature for *other* assessments. It is never used as a feature for its own row.

---

## 4. The pipeline, end to end

```
┌─────────────────────┐   ┌───────────────────────┐   ┌─────────────┐
│  assessments.csv    │   │ studentAssessment.csv │   │  vle.csv    │
│  (206 rows)         │   │   (per-student scores)│   │  (6364 rows)│
└──────────┬──────────┘   └──────────┬────────────┘   └──────┬──────┘
           │                         │                       │
           │      INNER JOIN         │  (drops 18 rows with  │  GROUP BY
           │  on id_assessment       │   no submissions)     │ (module,
           │                         │                       │  presentation)
           ▼                         ▼                       ▼
     ┌──────────────────────────────────────────┐     ┌─────────────────┐
     │ 188 assessments, each with its full      │     │ 3 aggregates    │
     │ list of student scores                   │     │ per module-pres │
     └────────────┬─────────────────────────────┘     └────────┬────────┘
                  │                                            │
                  │  compute fail_rate per assessment          │
                  │  risk_level = (fail_rate >= 0.10)          │
                  │  compute peer_fail_rate (leave-one-out)    │
                  │                                            │
                  ▼                                            │
     ┌─────────────────────────────────┐                      │
     │ 188 rows × 10 features + label  │                      │
     └────────────┬────────────────────┘                      │
                  │                                            │
                  │◄───  LEFT JOIN on (code_module, presentation)  
                  ▼
     ┌───────────────────────────────────────────┐
     │ Final modelling table                      │
     │   188 rows  ×  13 features  +  1 target    │
     └──────────────────────┬────────────────────┘
                            │
                            │ min-max scale continuous cols;
                            │ integer-encode categoricals
                            │
                            ▼
     ┌───────────────────────────────────────────┐
     │ Class-stratified 80/20 split (seed=42)    │
     │   Train: 151 rows   Test: 37 rows         │
     └──────────────────────┬────────────────────┘
                            │
                            │ train Logistic Regression from scratch
                            │ (1000 epochs, lr=0.1, class-weighted)
                            ▼
     ┌───────────────────────────────────────────┐
     │ Trained model: 13 weights + 1 bias        │
     └──────────────────────┬────────────────────┘
                            │
                            ▼
                      evaluate on test set
                    Accuracy / Precision / Recall / F1
```

---

## 5. The 13 features the model sees (inputs)

Grouped by source:

| Group | Source | Features |
|---|---|---|
| Assessment metadata (9) | `assessments.csv` | `module_id`, `presentation_id`, `assessment_type_id`, `date_norm`, `weight_norm`, `is_exam`, `is_high_weight`, `assessment_period`, `module_difficulty_score` |
| VLE catalog aggregates (3) | `vle.csv` | `n_vle_resources`, `n_activity_types`, `mean_resource_weeks` |
| Leave-one-out target encoding (1) | derived | `peer_fail_rate` |

**What the model does NOT see:** `fail_rate`, individual student scores, `risk_level`, `id_assessment`, `id_student`. Keeping these out of the feature set is what makes the prediction real rather than circular.

---

## 6. What the model outputs

For **any new assessment** (even one that hasn't happened yet), you feed the 13 features in and the model returns:

| Output | Form | Range | Meaning |
|---|---|---|---|
| `probability` | float | 0.00 – 1.00 | Estimated probability this assessment is High Risk |
| `risk_prediction` | integer | 0 or 1 | 1 if probability ≥ 0.5, else 0 |

Example single-row output (a TMA due late in a BBB presentation):
```
input  features  → [2, 3, 2, 0.71, 0.40, 0, 1, 2, 3, 188, 6, 12.4, 0.14]
sigmoid(w·x + b) → 0.82
risk_prediction  → 1   (High Risk — recommend coordinator review)
```

Across the entire held-out test set the outputs yielded:

| Metric | Value | Plain meaning |
|---|---|---|
| Accuracy | 78.38% | 29 of 37 test assessments classified correctly |
| Precision | 33.33% | When the model says "High Risk", 1 in 3 actually is |
| Recall | 100.00% | The model catches **every** truly High Risk assessment |
| F1-Score | 50.00% | Harmonic mean of precision and recall |
| Confusion matrix | TP=4, TN=25, FP=8, FN=0 | 4 risky assessments correctly flagged, 25 safe correctly cleared, 8 safe wrongly flagged, 0 risky missed |

The recall of 100% is the headline: as an early-warning system, missing a risky assessment is far worse than flagging a safe one for a 10-minute coordinator review.

---

## 7. One-sentence summary

> *We take OULAD's assessment metadata and online-resource catalog, label each of 188 assessments as High Risk (≥10% of students failed) using historical student scores that the model never sees as inputs, and train a from-scratch Logistic Regression to predict that label from 13 engineered features — so that a course coordinator can flag risky assessments in future presentations before any student sits them.*
