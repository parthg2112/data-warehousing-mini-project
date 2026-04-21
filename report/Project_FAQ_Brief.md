# Student Risk Prediction — Concept Brief & Faculty FAQ

A single-sheet reference for the **Data Warehousing & Mining** mini-project. Skim Part A for the elevator pitch; use Part B to answer the faculty's viva questions; Parts C and D cover the two "extras" the team was asked about; **Part E is the deep dive** — use it if you want the "why" behind the CSV choice and the LR math, in plain language.

---

## Part A — The project in one page

**Dataset.** We used the **Open University Learning Analytics Dataset (OULAD)** — a public dataset released by the UK Open University covering seven modules (AAA … GGG) across eight presentations between 2013J and 2014J. We pulled three of its CSV tables: `assessments.csv` (206 assessment records — module, presentation, type, due date, weight), `studentAssessment.csv` (per-student scores, used **only** to compute the target — not as an input feature), and `vle.csv` (6,364 online learning-resource records catalogued by module presentation). After an inner join on `id_assessment`, 188 assessments remained (18 rows without submissions — mostly end-of-module exams — were dropped).

**Problem framing.** **Binary classification at the assessment level.** Each of the 188 assessments is labelled `1` ("High Risk") if **≥ 10%** of students who attempted it scored below the OULAD pass mark of 40, and `0` otherwise. Class balance: 20 positives vs. 168 negatives (≈10.6% positive rate). The practical intent: let a course coordinator know *in advance* which assessments in an upcoming presentation are likely to be High Risk, so they can re-time it, reinforce related resources, or plan support.

**Pipeline.** `load CSVs → join → derive target → engineer 13 features → integer-encode categoricals and min-max scale numerics → class-stratified 80/20 split → train Logistic Regression from scratch with class-weighted gradient descent → evaluate at threshold 0.5`.

**Headline result.** Accuracy **78.38%**, Precision **33.33%**, Recall **100.00%**, F1 **50.00%** on 37 held-out assessments (TP=4, TN=25, FP=8, FN=0). Every High Risk assessment in the test set was caught. Precision is the trade-off: 8 "safe" assessments were also flagged for review. We report this as the **right** operating point for an early-warning system — a missed risky assessment hurts students; a flagged safe one just costs a coordinator 10 minutes of review.

---

## Part B — Faculty FAQ

### 1. Dataset kya hai?
**OULAD — Open University Learning Analytics Dataset.** Three tables used:

| Table | Rows used | Role |
|---|---|---|
| `assessments.csv` | 206 → 188 after join | Feature source: module, presentation, assessment type, due date, weight |
| `studentAssessment.csv` | per-student scores | **Label source only** — never an input feature. Used to compute `fail_rate` per assessment |
| `vle.csv` | 6,364 resources | Aggregated to 3 catalog features per assessment |

### 2. Predictive or classification?
**Classification — supervised binary.** Target derivation rule:
```
fail_rate = (# students scoring < 40) / (# submissions)
risk_level = 1 if fail_rate ≥ 0.10 else 0
```
20 positives, 168 negatives (imbalanced, ~10.6% positive prior).

### 3. Which algorithm?
**Logistic Regression.** Mechanics:

- Sigmoid: $\sigma(z) = 1/(1+e^{-z})$ where $z = w^\top x + b$.
- Loss: class-weighted Binary Cross-Entropy.
- Optimizer: batch gradient descent, lr = 0.1, 1000 epochs, weights initialised from `uniform(−0.01, 0.01)` (seed 42), bias = 0.
- Class weighting: positive-class errors scaled by `n_neg / n_pos ≈ 7.4` inside the gradient, so the minority class isn't ignored.
- Decision threshold: **0.5** on the sigmoid output.

### 4. Which code have you used?
**Pure Python in a Jupyter notebook — no ML library.** Breakdown of dependencies:

| Library | Used for |
|---|---|
| `pandas` | reading CSVs, joining tables, groupby aggregates |
| `matplotlib` | plots only |
| **Standard library** (`random`, `math`) | **all** of the ML: sigmoid, BCE loss, gradient step, prediction, confusion matrix, accuracy/precision/recall/F1 |

No scikit-learn, no NumPy arrays inside the model, no TensorFlow, no PyTorch. Every equation in Part B-3 above is hand-coded so the math is visible end-to-end.

### 5. What attributes are chosen?
**13 features in three groups.**

**Assessment metadata (9):**
- `module_id` — integer code for `code_module` (AAA…GGG)
- `presentation_id` — integer code for `code_presentation` (2013B, 2013J, 2014B, 2014J)
- `assessment_type_id` — CMA=0 / Exam=1 / TMA=2
- `date_norm` — due-day offset, min-max scaled to `[0, 1]`
- `weight_norm` — percentage contribution to final grade, min-max scaled
- `is_exam` — 1 if Exam, else 0
- `is_high_weight` — 1 if `weight > 20`, else 0
- `assessment_period` — 0 early / 1 mid / 2 late in presentation
- `module_difficulty_score` — coarse module-level ordinal difficulty

**VLE catalog aggregates (3):**
- `n_vle_resources` — number of online resources for this module presentation
- `n_activity_types` — distinct VLE activity types offered
- `mean_resource_weeks` — average "week available" of those resources

**Leave-one-out target encoding (1):**
- `peer_fail_rate` — for each row, the mean `fail_rate` of the *other* assessments in the same module (the row's own `fail_rate` is excluded from the average). This is a proper target-encoding trick: it leverages the target column without leaking the row's own label.

### 6. What preprocessing is done, if any?
1. **Inner join** of `assessments.csv` with `studentAssessment.csv` on `id_assessment` — drops 18 assessments with no submissions. Left join of VLE catalog aggregates on `(code_module, code_presentation)`.
2. **Target derivation** — `fail_rate` computed per assessment; thresholded at 0.10.
3. **Categorical encoding** — `code_module`, `code_presentation`, `assessment_type` → integer IDs (0…n−1). We deliberately did **not** one-hot encode: on 188 rows, integer codes keep the feature count low and the per-feature weight interpretable in the bar chart.
4. **Min-max scaling** applied to `date_norm`, `weight_norm`, and `peer_fail_rate` (the three continuous features with wide numerical ranges): `x' = (x − min) / (max − min + 1e−9)`. Binary / small-integer features are already in comparable scales, so we left them alone.
5. **Class-stratified 80/20 train/test split** (seed 42) — guarantees ~12% positive rate in both halves. Without this, a random split can put only 2 positives in the test set, which makes P/R/F1 resolution coarse and explains the artefactual 50% we saw in the initial baseline.
6. **Class weighting** — computed once on the training set as `n_neg / n_pos`.

### 7. Which algorithm has the best stats?
Only one algorithm is implemented (LR from scratch). The comparative story is an **ablation** across three configurations of the same model:

| Configuration | Accuracy | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| (a) Random split, no class weight | 94.74% | 50.00% | 50.00% | 50.00% | Artefact — only 2 test positives, so each prediction swings P/R by 50 pp |
| (b) Stratified split, no class weight | high | — | 0% | 0% | Model collapses to always predicting majority class — useless |
| **(c) Stratified + class-weighted (ours)** | **78.38%** | **33.33%** | **100.00%** | **50.00%** | Recall-first early-warning model — catches every TP |

We report **(c)** as best because the operating point matches the use case: **false negatives are far costlier than false positives** in an early-warning setting. A school coordinator reviewing 12 flagged assessments to catch all 4 genuinely risky ones is a good trade; missing 1 of those 4 is not.

---

## Part C — Random dummy variable (sanity check)

The notebook appends a single sanity-check cell that adds a **14th feature** — `noise_random`, drawn independently from `Uniform(0, 1)` (seeded, pure stdlib) — and retrains LR with identical hyperparameters. This is a classic diagnostic. If the model were merely memorising noise, random input would attract non-trivial weight and the test metrics would swing. If the model is learning real signal, the optimiser should shrink `noise_random`'s weight toward zero and the metrics should barely move.

**Measured result** (from the appendix cell `sanity_code_main` in the notebook):

| Quantity | Value |
|---|---|
| Learned weight on `noise_random` | **−0.277778** |
| `|noise_random|` | 0.277778 |
| Median `|weight|` across the 13 real features | 0.360656 |
| Max `|weight|` across the 13 real features | 3.728987 (peer_fail_rate) |
| Test Accuracy / Precision / Recall / F1 (14-feature) | 78.38% / 33.33% / 100.00% / 50.00% |
| Test Accuracy / Precision / Recall / F1 (headline, 13-feature) | 78.38% / 33.33% / 100.00% / 50.00% |
| Confusion matrix (14-feature) | TP = 4, TN = 25, FP = 8, FN = 0 |

- **Takeaway:** the noise weight is below the median real-feature weight *and* roughly **13× smaller** than the strongest real weight (`peer_fail_rate`). Every test-set metric is unchanged, down to the confusion matrix — adding a random column moved nothing. The printed sanity-check verdict is **PASS**.
- **What to quote in the viva:** "We added a uniformly random 14th feature and retrained. Its learned weight ended up at −0.28 — below the median of the real-feature weights and over ten times smaller than the top real feature. Every test metric stayed identical to the 13-feature model. That's direct evidence that our gradient descent is learning from the OULAD features, not memorising noise."
- **Where it lives:** the noise column is an appendix cell at the end of the notebook — the headline model, figures, and LaTeX report all remain on 13 features.

---

## Part D — Why Logistic Regression from scratch?

A fair question — sklearn's `LogisticRegression().fit(X, y)` would have been shorter. Four reasons we wrote it ourselves:

- **Course framing.** Data Warehousing & Mining is about *understanding* the mining algorithms, not treating them as black boxes. A from-scratch implementation forces the team to internalise sigmoid, BCE, the gradient derivation, and the update rule — which is exactly what gets examined.
- **Viva defensibility.** Every line in the notebook is something the team wrote and can whiteboard. "`model.fit(X, y)`" is one line that's very hard to defend under questioning; `for epoch in range(epochs): for xi, yi in ...` is 20 lines the team can narrate.
- **Ablation clarity.** The recall-first story depends on two small surgical edits — **class weighting** inside the gradient and **thresholding** at the sigmoid output. Both are 2–3 lines in our code and easy to explain. In sklearn they're hidden behind flags like `class_weight='balanced'` and `predict_proba + np.where`, which is harder to narrate without peeking at the library source.
- **Right-sized.** 188 rows × 13 features. The hand-coded loop converges in under a second. There is zero performance gain from a library, and plenty of pedagogical loss.

The reference paper the report is modelled on (`Breast_Cancer_IEEE_Revised.tex`) also implements its classifiers from scratch for the same reasons — we followed the same convention.

---

## Part E — Deep Dive

Study this section once; you won't need to reopen the notebook to answer follow-up questions.

### E1. The three CSVs — what each one *is for*, not just what it contains

Think of the three files as playing three different roles in the pipeline. Each one answers a different question about the world:

| File | Role | The question it answers |
|---|---|---|
| `assessments.csv` | **Unit of observation** | *"What assessment events exist, and what are their basic properties?"* |
| `studentAssessment.csv` | **Ground-truth oracle** | *"How did students actually do on each assessment?"* — used to **label** the data, never as input |
| `vle.csv` | **Context / environment catalog** | *"What online learning environment did students have available while preparing for this assessment?"* |

#### `assessments.csv` — one row per assessment event

This is the **spine** of the project. Every row here becomes one training example. Columns like `code_module` (which course, e.g. AAA, BBB…), `code_presentation` (which semester, e.g. 2013J), `assessment_type` (CMA / TMA / Exam), `date` (how many days into the semester it's due), and `weight` (% contribution to the final grade) are the **metadata features** — they describe the assessment itself. Nine of our 13 features are derived directly from this table.

#### `studentAssessment.csv` — one row per student submission

This file has `id_assessment`, `id_student`, and `score` (0 – 100). It's far larger than the assessment table because each assessment has many student submissions. **We never use its columns as features.** Instead, we use it exactly once to compute the label:

```
for each assessment_id:
    scores            = all scores for this assessment
    n_failed          = count of scores < 40          # OULAD pass threshold
    fail_rate         = n_failed / total submissions
    risk_level        = 1 if fail_rate >= 0.10 else 0
```

Why "label source only"? If we shipped this model to production and asked it to predict risk for a *new* 2015 presentation, we wouldn't have student scores yet — that's what the model is supposed to warn us about. Using `score` as an input feature would be **target leakage** — beautiful test metrics, useless model in production.

#### `vle.csv` — one row per online learning resource

OULAD's "VLE" (Virtual Learning Environment) is the Moodle-like system where students access videos, quizzes, forums, PDFs, etc. `vle.csv` catalogs 6,364 such resources, each tagged with the `code_module` and `code_presentation` it belongs to, an `activity_type` (resource / quiz / forum / oucontent / …), and the week range it's available. We don't care about individual resources; we care about the **shape of the learning environment around each assessment**. So we roll the table up into three numbers per (module, presentation):

- `n_vle_resources` — total resources available
- `n_activity_types` — distinct kinds of activity
- `mean_resource_weeks` — average availability window

These three numbers are the last three of our 13 features. They encode *"how resource-rich is the environment students had to prepare for this assessment?"* — a plausible predictor of fail-rate.

#### How they connect — the join diagram

```
                    +---------------------+
                    |   assessments.csv   |   (1 row per assessment event — 206 rows)
                    |  id_assessment  PK  |
                    |  code_module        |--------+
                    |  code_presentation  |--+     |
                    |  assessment_type    |  |     |
                    |  date, weight       |  |     |
                    +---------+-----------+  |     |
                              |              |     |
            INNER JOIN on     |              |     |    LEFT JOIN on
            id_assessment     |              |     |    (code_module,
                              |              |     |     code_presentation)
                              ▼              |     |            |
             +-----------------------------+ |     |            |
             | studentAssessment.csv       | |     |            |
             |   id_assessment  FK  ◄──────+ |     |            |
             |   id_student                | |     |            |
             |   score           ──► used  | |     |            |
             |                      to     | |     |            |
             |                      derive | |     |            |
             |                      LABEL  | |     |            |
             +-----------------------------+ |     |            |
                                             |     |            |
                                             |     ▼            ▼
                                             |  +---------------------------+
                                             |  |        vle.csv            |
                                             |  | code_module  FK           |
                                             |  | code_presentation  FK     |
                                             |  | activity_type             |
                                             |  | week_from, week_to        |
                                             |  +---------------------------+
                                             |       │
                                             |       │  GROUP BY
                                             |       │  (module, presentation)
                                             |       ▼
                                             |  aggregate → 3 numbers
                                             |  (n_vle_resources,
                                             |   n_activity_types,
                                             |   mean_resource_weeks)
                                             |
                                             +────► joined back onto assessments

Final modelling table: 188 rows × (13 features + 1 target)
```

- **Inner join** with `studentAssessment.csv` drops the 18 assessments that nobody ever submitted (mostly end-of-module exams recorded later) — we can't label them without submissions, so they're not useful for training.
- **Left join** with VLE aggregates — every assessment gets its module's environment context. If a module has no VLE entries (rare), we just get zeros, not a dropped row.
- The `peer_fail_rate` feature is computed *after* the `studentAssessment.csv` join, inside a group-by on `code_module`. It uses the label column but in a leave-one-out way (see Part B question 5), so each row's own label never appears in its own features — this is a valid "target encoding" trick, not leakage.

---

### E2. Logistic Regression in 5 minutes

The whole algorithm in one sentence: **take a weighted sum of the features, squash it to a number between 0 and 1, and interpret that number as the probability the row is High Risk.** Everything else — loss, gradient, training — is how we *pick the weights* to make those probabilities match the true labels.

#### Step 1: The linear score

For a row with features $x = (x_1, x_2, \dots, x_{13})$:

$$
z = w_1 x_1 + w_2 x_2 + \dots + w_{13} x_{13} + b = w^\top x + b
$$

`z` is any real number (could be -7.2, could be +4.8). `w_j` is the learned weight telling us *how much* feature `j` pushes the prediction toward High Risk; `b` (bias) is a baseline offset.

#### Step 2: Squash to a probability — the sigmoid

Linear scores live in $(-\infty, +\infty)$; probabilities live in $[0, 1]$. Sigmoid is the bridge:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Three numbers to remember:

| $z$ | $\sigma(z)$ | Meaning |
|---|---|---|
| $-\infty$ | 0.00 | "definitely Low Risk" |
| 0 | 0.50 | "no idea, coin-flip" |
| $+\infty$ | 1.00 | "definitely High Risk" |

Shape: a smooth S-curve. Small changes in `z` near 0 cause big changes in probability; far from 0 the curve saturates.

#### Step 3: Decide

$$
\hat{y} = 1 \text{ if } \sigma(z) \geq 0.5, \text{ else } 0
$$

A threshold of 0.5 on sigmoid output is the same as a threshold of 0 on the raw linear score `z`. Simple.

#### Step 4: Score the guess — Binary Cross-Entropy (BCE) loss

We need a number that tells us *how wrong* a single prediction was. For true label $y \in \{0, 1\}$ and predicted probability $\hat{p} = \sigma(z) \in (0, 1)$:

$$
L(\hat{p}, y) = -\bigl[\,y \log \hat{p} + (1 - y) \log(1 - \hat{p})\,\bigr]
$$

Why this formula? Plug in extreme cases:

| True $y$ | Predicted $\hat{p}$ | $L$ | Comment |
|---|---|---|---|
| 1 | 0.99 | $-\log(0.99) \approx 0.01$ | Confident, correct → tiny penalty |
| 1 | 0.01 | $-\log(0.01) \approx 4.6$ | Confident, wrong → huge penalty |
| 0 | 0.01 | $-\log(0.99) \approx 0.01$ | Confident, correct |
| 0 | 0.99 | $-\log(0.01) \approx 4.6$ | Confident, wrong |

BCE punishes **confident wrong answers** much harder than unsure ones. The total training loss is the average of `L` across all training rows.

#### Step 5: Find the weights — gradient descent

We want weights that make the total BCE loss as small as possible. Gradient descent is a blind-hiker algorithm: *at each step, feel which direction points downhill, take a small step that way, repeat*. "Downhill" = direction opposite to the gradient of the loss.

The magical simplification for logistic regression + BCE: after the calculus shakes out, the gradient for a single row is

$$
\frac{\partial L}{\partial w_j} = (\hat{p} - y)\,x_j \qquad \frac{\partial L}{\partial b} = (\hat{p} - y)
$$

That's it — "**error × input**". The same elegant form as linear regression. Average over all training rows, then update:

$$
w_j \leftarrow w_j - \eta \cdot \frac{1}{m}\sum_{i=1}^{m} (\hat{p}_i - y_i)\,x_{ij}
$$

$\eta$ = learning rate = 0.1 (step size). We repeat this 1000 times over the whole training set (1000 "epochs"), and the weights settle into values that minimize the loss. The notebook prints the loss every 100 epochs as a sanity check that it's decreasing — that's the "learning curve" figure.

#### Step 6: Fix the imbalance — class weighting

Problem: only ~12% of training rows are High Risk. Plain gradient descent gets dominated by the 88% negatives and converges to "always predict Low Risk", which has 88% accuracy but catches zero risky assessments. Useless.

Fix: scale every positive-sample error by `pos_weight = n_neg / n_pos ≈ 7.4`:

$$
\text{error}_i =
\begin{cases}
  (\hat{p}_i - y_i) \cdot 7.4 & \text{if } y_i = 1 \\
  (\hat{p}_i - y_i) \cdot 1.0 & \text{if } y_i = 0
\end{cases}
$$

Effect: each positive row now contributes 7.4× as much to the gradient as a negative row, so the total "pull" from positives equals the total "pull" from negatives. The model stops ignoring the minority class and starts finding them. This is the single change that turned Recall from 0% into 100%.

#### Step 7: Hyperparameters, short version

- **Learning rate = 0.1.** Moderate. Too big → weights oscillate and loss diverges; too small → takes forever.
- **1000 epochs.** One full pass through the training set = one epoch. We ran 1000; the loss curve flattens well before that, so we're not under-trained.
- **Weights init ~ Uniform(−0.01, +0.01), seed = 42.** Starting at exactly zero would leave all features perfectly symmetric and gradient descent couldn't break the symmetry; a tiny random seed breaks it. Fixed seed = reproducible runs.
- **Bias init = 0.** No prior preference.
- **Threshold = 0.5.** Our class weighting already rebalances the posterior, so 0.5 is the natural cut-off. (If we hadn't class-weighted, we'd have to tune the threshold instead — both achieve the same kind of fix.)

#### The whole algorithm, pseudocode-style

```
initialise weights w (tiny random), bias b = 0
for epoch in 1..1000:
    total_dw = 0, total_db = 0
    for each training row (x, y):
        z     = w·x + b
        p     = sigmoid(z)
        err   = (p - y) * (7.4 if y==1 else 1.0)   # class weight
        total_dw += err * x                         # accumulate gradient
        total_db += err
    w -= 0.1 * total_dw / m_effective               # single update per epoch
    b -= 0.1 * total_db / m_effective

# At inference time:
predict(x) = 1 if sigmoid(w·x + b) >= 0.5 else 0
```

That is the entire learning algorithm. Twenty lines. The notebook's version has the same structure; the from-scratch implementation is deliberately readable so you can point to any line in the viva and say exactly what it does.

---

## Part F — Clearing up a common misconception: "Derived labels aren't real labels, use k-means instead"

This section directly addresses the faculty objection:

> *"Logistic Regression can only be used for classification, and for classification there must already be a label column in the dataset listing High Risk / Low Risk. Since you derived the label from a rule (`fail_rate >= 0.10`), this is not a supervised problem — you should have used an unsupervised algorithm like K-Means."*

There are **three separate mistakes tangled together** in that argument. Untangling them:

### F1. The label does **not** have to be pre-printed in the CSV

Supervised learning needs one thing: **a known target value for every training row at training time**. It is completely silent on where that value came from. The target can be:

| Source of the label | Example |
|---|---|
| Manually annotated by a human | ImageNet (humans tag each image) |
| Captured from user behaviour | Spam filter (users click "mark as spam") |
| Read from a ground-truth record later | Credit default (did the borrower repay within 90 days? — known only in hindsight) |
| Diagnosed by an expert | Breast-cancer benign/malignant (pathologist reviews the biopsy) |
| **Computed from raw data by a rule** | **Our project: `fail_rate ≥ 0.10`** |

The last row is how most real-world ML systems work. You almost never ship a CSV where column 14 happens to be the exact label you want. You derive it. Spam filters don't come with a `spam/ham` column baked in — teams *construct* the label by aggregating user clicks. Nobody looks at Gmail and says "that's unsupervised because the dataset didn't arrive with spam labels attached." The label-derivation step is called **label engineering**, and it's a completely standard, well-documented phase of any supervised pipeline.

**The rule for distinguishing supervised from unsupervised is simple:**

> At training time, do I know the correct answer for each row — yes or no?

- *Yes* → supervised (classification or regression).
- *No* → unsupervised (clustering, dimensionality reduction, anomaly detection).

We compute `risk_level` for all 188 rows **before training**. So we know the answer for every row. Therefore: supervised.

### F2. K-Means would actively make the problem worse, not solve it

K-Means is a **clustering** algorithm. Clustering's purpose is to find structure in data that has **no** labels. The output of K-Means is "this row belongs to group 2" — where group 2 has no semantic meaning by itself. K-Means has no concept of "High Risk". It has no `y`, no target, no right answer to aim for.

If we ran K-Means on our 13 features with `k = 2`, we would get two clusters, but:

1. **We don't know which cluster means "High Risk".** We'd have to manually inspect cluster centroids and *assign a label* — which is exactly the rule-based labelling step the faculty claims is disqualifying. We haven't avoided the rule; we've just pushed it further down the pipeline, without the benefit of a ground-truth target during training.
2. **The clusters would optimise geometric distance, not risk prediction.** K-Means minimises within-cluster Euclidean distance. There is no reason to believe the "close in feature space" partition matches the "assessments where ≥10% of students fail" partition. The two objectives are unrelated.
3. **We'd lose the ability to measure accuracy, precision, recall, or F1.** Those metrics require comparing predictions against a known truth. K-Means gives us no truth to compare against, so we couldn't even prove the model works.
4. **At deployment, we still couldn't predict new assessments cleanly.** A new 2025J assessment would get placed in the nearest cluster, but that's geometric proximity, not "probability this fails". We wouldn't have a probability at all.

K-Means is the right tool when you genuinely have no idea what the groups are and want the data to speak (e.g. customer segmentation, exploratory analysis). It's the wrong tool when you already know the outcome you care about (High Risk vs. Low Risk) and just need a model that predicts it.

### F3. The "circularity" worry — are we just predicting our own rule?

This is the most sophisticated version of the objection, and it deserves an honest answer: *"If the label is just `fail_rate ≥ 0.10`, isn't the model trivially predicting something we already computed?"*

**No, because the model never sees `fail_rate`.**

Look carefully at the pipeline:

- `fail_rate` is computed once, from `studentAssessment.csv`, and is used **only** to set `risk_level`.
- The 13 input features come from **`assessments.csv` and `vle.csv` alone** (plus the leave-one-out `peer_fail_rate`, which never contains the row's own fail-rate). **None of these are `fail_rate`**. None of them is derived from the row's own student scores.
- So at prediction time the model sees only: *module, presentation, assessment type, date, weight, a few binary flags, three VLE aggregates, and the fail-rates of other assessments in the same module*. From those, it has to guess whether **this** assessment will have fail-rate ≥ 10%.
- That is a **genuine predictive problem.** The rule `fail_rate ≥ 0.10` tells us the **answer**; it is not one of the **inputs**. Learning to predict the answer from inputs that don't contain it is the whole point of supervised learning.

The concrete production scenario makes this obvious. Imagine a 2025J presentation being planned today. We have every one of our 13 features for each proposed assessment (the module, the presentation code, the type, the due date, the weight, the VLE catalog that will be provided, and the historical peer fail-rate in the same module). We do **not** have student scores, because students haven't sat the assessment yet — that's the whole reason we need the model. Our LR takes the 13 features and outputs a probability of High Risk. That probability is a real prediction about a future event, even though the label it was trained on was computed from historical scores.

This is identical in structure to medical risk prediction: a radiologist labels 10,000 X-rays as "cancer / no cancer" using biopsy results we only have for *past* patients; we train a classifier on the images (not the biopsies); at deployment the classifier sees a new X-ray (no biopsy yet) and outputs a probability. The label comes from a downstream ground-truth source that doesn't exist at prediction time. Exactly our setup.

### F4. When the faculty's objection *would* be correct

To be fair to the faculty position, there are two scenarios where her critique would land:

1. **If we had used `fail_rate` itself as a feature.** Then the model would trivially learn "if `fail_rate ≥ 0.10` → output 1, else 0" with 100% accuracy. That *would* be circular, and it's precisely the target-leakage trap we explicitly avoid — see Section V.B of the LaTeX report.
2. **If we had no way to compute labels at all** — e.g. if `studentAssessment.csv` simply didn't exist. Then supervised learning would be off the table and we'd be forced into clustering or some other unsupervised method. But we do have `studentAssessment.csv`, so this branch never applies.

Neither of those scenarios is what we did, so neither objection bites.

### F5. Viva-ready rebuttal (one paragraph to deliver calmly)

> *Ma'am, supervised learning requires that we have a known target value for every training row — it does not require the target to be pre-printed in the CSV. In real-world ML this is almost never the case. Spam filters derive their "spam" label from user behaviour, credit models derive "default" from hindsight records, medical classifiers derive their label from later biopsies. In our project we derive `risk_level` once, from `studentAssessment.csv`, using a published OULAD pass-mark rule — that's called **label engineering** and it's a standard pipeline step. Once derived, we have 188 rows each with a known label, which by definition makes it a supervised classification problem. K-Means would be wrong here because K-Means has no target, no measurable accuracy, and its clusters would optimise geometric distance rather than "will ≥10% of students fail this assessment". Critically, our 13 input features never include `fail_rate` itself, so the model is not predicting its own rule — it's predicting a future outcome from assessment metadata and environmental context that would also be available at deployment, when student scores don't yet exist. This is the same structure as every production classifier built on top of derived labels.*

Polite, grounded, and correct. If pushed further, walk through Part F1–F4 one at a time.

---

## One-line summary for the viva panel

> *We built a binary-classifier pipeline on the OULAD dataset that flags High Risk assessments before a course presentation begins, using 13 engineered features (assessment metadata + VLE catalog aggregates + a leakage-free peer fail-rate) and a class-weighted Logistic Regression we implemented from scratch in pure Python. On a stratified test split, it catches 100% of the risky assessments at the cost of some reviewable false alarms — the correct operating point for an early-warning system.*
