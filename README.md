# Student Risk Prediction using Logistic Regression

Predicts whether a student assessment is high-risk or low-risk using Logistic Regression
implemented entirely from scratch (no ML libraries), on the Open University Learning
Analytics Dataset (OULAD). Built as a Data Warehousing and Mining mini-project.

## Features

- Risk classification of assessments from metadata: weighting, due-date timing, assessment type (TMA, CMA, Exam)
- Logistic Regression (with Decision Tree and Random Forest baselines) implemented from scratch — no scikit-learn
- Class-imbalance handling via class-weighted training (`pos_weight = n_neg/n_pos`), recall-first
- Hand-rolled evaluation: Accuracy, Precision, Recall, F1, confusion matrix, learning curve
- A noise-feature control experiment validating that learned weights are not spurious
- 17 diagnostic figures and a full project report (`report/Student_Risk_DWM_Report.pdf`)

## Tech Stack

Python, pandas, NumPy, Matplotlib, Jinja2, Jupyter Notebook

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook notebooks/student_risk_prediction_logistic_regression.ipynb
```

The OULAD CSVs (`assessments.csv`, `courses.csv`, `studentAssessment.csv`, `studentInfo.csv`,
`studentRegistration.csv`, `vle.csv`) ship in `data/`.

## Results

All three classifiers are judged on the same 37 test rows (13 features, stratified 80/20
split), so differences reflect the algorithms themselves. In an early-warning setting the
priority is recall, then F1, then precision.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (from scratch) | 78.38% | 33.33% | 100.00% | 50.00% |
| Decision Tree (CART, from scratch) | 97.30% | 80.00% | 100.00% | 88.89% |
| Random Forest (from scratch) | 94.59% | 75.00% | 75.00% | 75.00% |

- Notebook HTML/PDF exports: `exports/`
- Full report: `report/Student_Risk_DWM_Report.pdf`
- Figures: `figures/`

## License

MIT — see [LICENSE](LICENSE).
