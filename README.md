# Student Risk Prediction using Logistic Regression

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Predicting student assessment risk levels using Logistic Regression implemented from scratch. This project utilizes the **Open University Learning Analytics Dataset (OULAD)** to identify high-risk assessments based on weighting, timing, and assessment type.

## 🚀 Overview

Educational institutions can use predictive analytics to flag at-risk students before performance deteriorates. This notebook demonstrates:
- **Feature Engineering**: Deriving risk signals from assessment metadata.
- **Mathematical Implementation**: Logistic Regression built from scratch (no ML libraries).
- **Evaluation**: Performance metrics including Accuracy, Precision, Recall, and F1-Score.

## 📁 Project Structure

```text
.
├── data/               # Raw OULAD dataset files (CSV)
├── notebooks/          # Jupyter notebooks with analysis
├── exports/            # Exported reports (HTML, PDF)
├── requirements.txt    # Project dependencies
├── .gitignore          # Git exclusion rules
└── README.md           # Project documentation
```

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd dwm_project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis**:
   ```bash
   jupyter notebook notebooks/student_risk_prediction_logistic_regression.ipynb
   ```

## 📊 Dataset

The project uses the **OULAD** dataset. The primary file analyzed is `assessments.csv`, containing:
- `code_module`: Module identifier.
- `assessment_type`: TMA, CMA, or Exam.
- `date`: Due date day.
- `weight`: Percentage weight in final grade.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
