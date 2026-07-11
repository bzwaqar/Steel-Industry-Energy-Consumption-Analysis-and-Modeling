# ⚡ Steel Industry Energy Consumption — Advanced EDA & Modeling

> Multi-class classification of `Load_Type` (Light / Medium / Maximum) using electrical
> consumption parameters, with a production-style feature engineering, imbalance-handling,
> and explainability pipeline.

---

## 📌 Project Overview

This project analyzes 15-minute interval energy consumption data from a steel industry
facility to:

1. Perform deep **Exploratory Data Analysis (EDA)** — including statistical outlier
   detection, mutual information scoring, and time-series trend analysis.
2. Build an **advanced multi-model classification pipeline** to predict `Load_Type`,
   handling class imbalance and comparing linear, ensemble, and gradient-boosted models.
3. Explain model predictions using **SHAP (SHapley Additive exPlanations)**.

This is an upgraded version of a baseline EDA + 3-model project — rebuilt with
industry-style practices (leak-free pipelines, cyclical time encoding, cross-validation,
hyperparameter search, and explainability).

---

## 📂 Repository Structure

```
├── data/
│   └── Week 2 (DataSet).xlsx
├── README.md
├── requirements.txt
├── week2_eda_advanced.ipynb
└── week2_advanced_models.ipynb
```

---

## 📊 Dataset Information

| | |
|---|---|
| **Source** | Steel Industry Energy Consumption dataset |
| **Granularity** | 15-minute interval readings, 1 year |
| **Target** | `Load_Type` — Light_Load / Medium_Load / Maximum_Load |
| **Key Features** | `Usage_kWh`, `Lagging/Leading_Reactive_Power`, `CO2`, `Lagging_Current_Power_Factor`, `NSM`, `WeekStatus`, `Day_of_week` |

---

## ⚙️ Environment Setup

```bash
git clone <your-repo-url>
cd steel-industry-energy-consumption

# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Google Colab users:** if `xgboost`, `lightgbm`, or `shap` aren't pre-installed, run this
> in the first cell:
> ```
> !pip install xgboost lightgbm shap imbalanced-learn -q
> ```

---

## 🧠 Feature Engineering (Advanced)

- Parsed `date` into **cyclical features** (`hour_sin`, `hour_cos`, `month_sin`,
  `month_cos`) so the model understands that 23:00 and 00:00 are close together —
  something raw numeric hour encoding cannot capture.
- Extracted `is_weekend`, `weekday` as additional temporal signals.
- One-Hot Encoded `WeekStatus` and `Day_of_week`.
- Label Encoded the target `Load_Type`.
- Standard-scaled all numeric features **inside the pipeline** (fit only on training folds
  — no data leakage).

---

## 🔍 EDA Highlights (`week2_eda_advanced.ipynb`)

- **Data quality audit** — missing %, duplicates, dtypes in one summary table.
- **IQR-based outlier detection** per feature (quantified, not just visual).
- **Mutual Information scoring** — captures non-linear feature-target relationships that
  correlation alone misses.
- **7-day rolling average trend** of energy usage over the year.
- Violin plots for Power Factor vs Load Type (domain-relevant relationship).

---

## 🤖 Modeling Pipeline (`week2_advanced_models.ipynb`)

| Step | What's used | Why it matters |
|---|---|---|
| Imbalance handling | `SMOTE` inside an `imblearn` Pipeline | Prevents leakage — resampling happens only on training folds |
| Models compared | Logistic Regression, Random Forest, **XGBoost**, **LightGBM** | Linear baseline → ensemble → gradient boosting |
| Validation | 5-fold **Stratified Cross-Validation** (F1-macro) | Reliable metric for imbalanced multi-class problems |
| Tuning | `RandomizedSearchCV` (20 iterations) | Fast, near-exhaustive hyperparameter search |
| Explainability | **SHAP** summary plots | Shows *why* the model made each prediction, not just accuracy |
| Extra metric | Macro **ROC-AUC** (One-vs-Rest) | Measures class separation quality beyond hard predictions |

### Results Summary
- **XGBoost (tuned)** achieved the best F1-macro and ROC-AUC among all models.
- SMOTE meaningfully improved recall on the minority `Maximum_Load` class.
- SHAP confirmed **Reactive Power** and **Power Factor** as the top decision-driving
  features — consistent with electrical engineering domain knowledge.

---

## 🚀 Possible Next Steps

- Bayesian hyperparameter optimization with **Optuna**.
- Time-series modeling (LSTM/Prophet) since the data is naturally sequential.
- Deploying the tuned model as a REST API via **FastAPI**.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `imbalanced-learn` · `XGBoost` · `LightGBM` ·
`SHAP` · `seaborn` / `matplotlib`

---

## 👤 Author

**Waqar** — AI/ML Engineering Student, COMSATS University Islamabad
[GitHub](https://github.com/bzwaqar) · [LinkedIn](https://linkedin.com/in/waqar-khan-9a7016321)
