# Student Risk Prediction — Readable Script + Faculty FAQ

---

## Part 0 — The 2-page explainer (read this first)

### What we built

A small pipeline that looks at an OULAD course assessment (module, type, due date, weight, the online resources around it, and how risky the other assessments in the same module were) and predicts whether that assessment will be a **High Risk** event — meaning at least 10% of students are likely to fail it. A course coordinator can then re-time it, strengthen the related VLE resources, or plan support *before* students sit it.

### The dataset in one paragraph

We used the **Open University Learning Analytics Dataset (OULAD)**. Three of its CSVs feed the pipeline:

- `assessments.csv` — 206 rows, one per assessment event (module, presentation, type, due date, weight). This is the **spine**: each row becomes one training example.
- `studentAssessment.csv` — per-student scores. Used **only** to compute the label (never as input). For each assessment: `fail_rate = fraction scoring < 40`, then `risk_level = 1 if fail_rate ≥ 0.10 else 0`.
- `vle.csv` — 6,364 online learning resources. Rolled up into three catalog aggregates per (module, presentation).

After an inner join that drops 18 assessments with no submissions, **188 rows** remain. Class balance ≈ **11.7% positive** (22 High Risk out of 188) — imbalanced, and that drives several design choices below.

### What the model sees (13 features)

- **9 assessment-metadata features** from `assessments.csv`: module id, presentation id, type code, normalised due-date, normalised weight, `is_exam`, `is_high_weight`, `assessment_period`, `module_difficulty_score`.
- **3 VLE aggregates** from `vle.csv`: `n_vle_resources`, `n_activity_types`, `mean_resource_weeks`.
- **1 peer signal**: `peer_fail_rate` — the mean fail-rate of the *other* assessments in the same module (leave-one-out, so a row never sees its own label).

The model **never** sees the student scores or `fail_rate` itself. That's what makes the prediction real rather than circular.

### Why binary classification — and why our dataset fits it

The question "will this assessment be risky?" has two outcomes: yes or no. That is binary classification by definition. We can frame it this way because `studentAssessment.csv` lets us compute a ground-truth label (`risk_level`) for every past assessment, so at training time every row has a known answer. That satisfies the core requirement of supervised learning. At deployment, a new assessment being planned doesn't have student scores yet (that's the point of the warning system), but it does have all 13 features — module, type, due date, weight, VLE catalog, and historical peer fail-rate in the same module. So the model can predict risk for assessments that haven't happened yet.

Regression doesn't fit here because we don't need to predict the exact fail-rate — a coordinator just needs a yes/no flag. Unsupervised clustering (e.g. K-Means) doesn't fit because we already know what the groups mean and we want measurable accuracy/recall against a ground truth.

### Why Logistic Regression (and why from scratch)

LR is the textbook baseline for binary classification: take a weighted sum of the 13 features, squash it through a sigmoid to get a probability, threshold at 0.5. It's small enough to implement in ~20 lines of pure Python, and every equation (sigmoid, binary cross-entropy, gradient, update) is visible end to end. That fits the course — Data Warehousing & Mining is about *understanding* mining algorithms, not calling `sklearn.fit`. It also makes the two key surgical edits easy to point at in the viva:

1. **Class weighting** — positive-sample errors are scaled by `n_neg/n_pos ≈ 7.4` inside the gradient, so the model stops ignoring the 12% minority class. This is the single change that moved recall from 0% to 100%.
2. **Stratified 80/20 split (seed 42)** — guarantees ~12% positives in both train and test. Without it, the tiny test set can have only 2 positives, which makes every metric jump in 50-percentage-point steps.

**Headline result for LR:** Accuracy 78.38%, Precision 33.33%, **Recall 100.00%**, F1 50.00% on 37 held-out assessments (TP=4, TN=25, FP=8, FN=0). Every risky assessment was caught. Eight safe ones were also flagged — a manageable review load in exchange for not missing any real ones.

### The two additional models (Decision Tree and Random Forest)

To give the comparison some depth, we added two more classifiers, again from scratch and with the same class-weighted setup so the comparison is apples-to-apples.

- **Decision Tree (CART)** — splits the feature space along axis-aligned thresholds using a weighted Gini impurity. Root split lands on `peer_fail_rate` at 0.49, confirming that the peer signal is the dominant axis. Result: **97.30% / 80.00% / 100.00% / 88.89%**. It keeps LR's perfect recall but cuts false positives from 8 down to 1.
- **Random Forest** — 50 trees, each trained on a bootstrap sample with 3 random features per split, predictions averaged. Result: 94.59% / 75.00% / 75.00% / 75.00%. Loses one borderline positive to the variance-reducing averaging, which is the expected bagging trade-off on such a small test set.

**What all three models agree on:** `peer_fail_rate` is the single most informative feature. That cross-algorithm agreement is a sanity check — the signal is real, not an artefact of any one method.

**What we ship:** the Decision Tree. Perfect recall, tight precision, simple to explain.

### One-line summary

> *We built an early-warning pipeline for OULAD assessments: 13 leakage-safe features per assessment, three from-scratch classifiers (LR, DT, RF) trained with matched class weighting, evaluated on a stratified test split. The Decision Tree is the strongest (97% accuracy, 100% recall); all three agree that the peer fail-rate of neighbouring assessments in the same module is the dominant predictor.*

---

## Part A — Faculty FAQ (concise)

### 1. Dataset?

OULAD. Three tables. `assessments.csv` (206 → 188 after join) feeds features. `studentAssessment.csv` feeds the label only. `vle.csv` (6,364 resources) rolls up to 3 aggregates per module-presentation.

### 2. Predictive or classification?

Supervised binary classification. Label: `risk_level = 1 if fail_rate ≥ 0.10 else 0`. Class balance: ~12% positive.

### 3. Which algorithms?

Three, all from scratch in pure Python:

- **Logistic Regression** — sigmoid + binary cross-entropy, batch gradient descent (lr 0.1, 1000 epochs), class-weighted, threshold 0.5.
- **Decision Tree (CART)** — weighted Gini, `max_depth=5`, `min_samples_split=4`.
- **Random Forest** — 50 trees, bootstrap samples, `√13 ≈ 3` features per split, probability averaging.

### 4. Code / dependencies?

`pandas` for CSV I/O and joins, `matplotlib` for plots. Every ML line (sigmoid, BCE, gradients, trees, forest, metrics) is pure stdlib. No scikit-learn, no NumPy inside the model, no TF/PyTorch.

### 5. 13 features?

**Metadata (9):** `module_id`, `presentation_id`, `assessment_type_id`, `date_norm`, `weight_norm`, `is_exam`, `is_high_weight`, `assessment_period`, `module_difficulty_score`.
**VLE aggregates (3):** `n_vle_resources`, `n_activity_types`, `mean_resource_weeks`.
**Peer signal (1):** `peer_fail_rate` — leave-one-out mean fail-rate of other assessments in the same module.

### 6. Preprocessing?

1. Inner join on `id_assessment` (drops 18 rows without submissions). Left join VLE aggregates on `(code_module, code_presentation)`.
2. Derive `fail_rate`, threshold at 0.10 to get `risk_level`.
3. Integer-encode `code_module`, `code_presentation`, `assessment_type`.
4. Min-max scale continuous features (`date_norm`, `weight_norm`, `peer_fail_rate`).
5. Class-stratified 80/20 split (seed 42).
6. Compute `pos_weight = n_neg/n_pos ≈ 7.4` on training data.

### 7. Which model performed best?

| Model | Acc | Prec | Rec | F1 |
|---|---|---|---|---|
| Logistic Regression | 78.38% | 33.33% | **100%** | 50.00% |
| **Decision Tree** | **97.30%** | **80.00%** | **100%** | **88.89%** |
| Random Forest | 94.59% | 75.00% | 75.00% | 75.00% |

Decision Tree wins on three metrics, ties LR on recall. All three agree `peer_fail_rate` is the top feature.

### 8. Random-dummy sanity check?

We added a 14th feature drawn from `Uniform(0, 1)` and retrained LR. Its learned weight settled at −0.28 (below the median real-feature weight, ~13× smaller than `peer_fail_rate`). Every test metric stayed identical. Verdict: the optimiser learns from real features, not noise.

---

## Part B — Common misconception: "derived labels aren't real, use K-Means instead"

Three things tangled together in this objection:

**B1. Labels don't have to be pre-printed in the CSV.** Supervised learning requires a known label at training time — it's silent on where that label came from. Spam filters derive the label from user clicks. Credit models derive "default" from hindsight. Medical classifiers derive "cancer" from later biopsies. Deriving `risk_level` from `studentAssessment.csv` via a published OULAD rule is called **label engineering** and is a standard pipeline step. We know the answer for all 188 rows before training → supervised.

**B2. K-Means would make things worse.** It has no target, no concept of "High Risk", and its clusters optimise Euclidean distance, not risk. We'd still have to label the clusters by hand afterwards (same rule-based step, just pushed further down), and we'd lose any way to measure accuracy/precision/recall.

**B3. Not circular — the model never sees `fail_rate`.** The 13 features come from `assessments.csv` and `vle.csv` plus the leave-one-out peer signal. `fail_rate` is used exactly once, to set the label. At deployment a new 2025J assessment has all 13 features but no student scores yet — which is exactly why the model is useful.

**When the objection would land:** if we used `fail_rate` itself as a feature (we don't — that's the leakage trap we explicitly avoid), or if `studentAssessment.csv` didn't exist at all (it does).

### Viva-ready rebuttal (one paragraph)

> *Supervised learning needs a known label at training time, not a pre-printed one. In real ML, labels are almost always derived — spam from user clicks, default from hindsight, cancer from later biopsies. We derive `risk_level` once from `studentAssessment.csv` via the OULAD pass-mark rule; that's label engineering, a standard step. Our 13 input features never include `fail_rate`, so the model predicts a future outcome from metadata and context that will also be available at deployment when student scores don't yet exist. K-Means would be wrong because it has no target, no measurable accuracy, and optimises geometric distance rather than risk.*

---

## Part C — Logistic Regression in 5 minutes (reference)

**Linear score:** $z = w^\top x + b$.
**Sigmoid:** $\sigma(z) = 1/(1+e^{-z})$ maps $z$ to a probability in $(0,1)$.
**Decision:** $\hat{y}=1$ if $\sigma(z) \ge 0.5$.
**Loss (BCE):** $L = -[y \log \hat{p} + (1-y)\log(1-\hat{p})]$ — punishes confidently-wrong answers hardest.
**Gradient:** $\partial L/\partial w_j = (\hat{p}-y)x_j$ and $\partial L/\partial b = (\hat{p}-y)$ — "error × input".
**Update:** $w \leftarrow w - \eta \cdot \frac{1}{m}\sum (\hat{p}_i - y_i)x_i$ with $\eta=0.1$, 1000 epochs.
**Class weighting:** positive errors scaled by `pos_weight ≈ 7.4` so the minority class isn't steamrolled.

```
initialise w small-random, b = 0
for epoch in 1..1000:
    for (x, y) in training data:
        p   = sigmoid(w·x + b)
        err = (p - y) * (7.4 if y==1 else 1.0)
        accumulate err·x into dw, err into db
    w -= 0.1 * dw / m;  b -= 0.1 * db / m
predict(x) = 1 if sigmoid(w·x + b) >= 0.5 else 0
```

That's the whole learning algorithm — 20 lines, nothing hidden.
