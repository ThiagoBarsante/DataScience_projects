# Local coding agent — Analysis & Prediction

End-to-end analysis of the [Telco Customer Churn](../data/) dataset: exploratory analysis, model comparison, hyperparameter tuning, and explainability, with actionable retention recommendations. All work lives in one executed notebook:

**[`customer_churn_analysis.ipynb`](customer_churn_analysis.ipynb)**

- **Dataset:** 7,043 customers × 21 columns
- **Task:** binary classification — predict whether a customer will churn (`Churn` = Yes/No)
- **Overall churn rate:** 26.5%

---

## Key findings

Churn is concentrated and explainable. Three levers account for most of it:

| Driver | Insight |
|---|---|
| **Contract type** | Month-to-month customers churn **42.7%** vs. 11.3% (one-year) and **2.8%** (two-year) |
| **Tenure** | New customers (first 0–6 months) are by far the highest-risk group; churn drops steadily with tenure |
| **Services & support** | Fiber-optic internet, electronic-check payment, and *lack* of TechSupport / OnlineSecurity all associate with higher churn |

Permutation importance on the final model confirms the same ranking — **Contract** and **tenure** dominate, followed by internet service, online security, and charges. These are levers the business controls, not just correlations.

---

## Model comparison

Five algorithms were trained inside a shared, leakage-safe preprocessing pipeline and scored on a held-out 20% test set (stratified). Class imbalance is handled with `class_weight="balanced"` / `scale_pos_weight`.

| Model | ROC-AUC | Recall (churn) | Precision (churn) | F1 (churn) | Accuracy |
|---|---|---|---|---|---|
| **Gradient Boosting** ✅ | **0.843** | 0.51 | 0.67 | 0.58 | 0.80 |
| Logistic Regression | 0.842 | 0.78 | 0.50 | 0.61 | 0.74 |
| XGBoost | 0.841 | 0.78 | 0.53 | 0.63 | 0.75 |
| LightGBM | 0.839 | 0.77 | 0.52 | 0.62 | 0.75 |
| Random Forest | 0.822 | 0.64 | 0.55 | 0.59 | 0.77 |

The models are tightly clustered on ROC-AUC. Note the trade-off: the boosted trees give the best *ranking* (ROC-AUC) but predict conservatively at the default 0.5 threshold (lower recall), while Logistic Regression / XGBoost / LightGBM catch ~78% of churners at the cost of more false alarms.

---

## Understanding the metrics

All three metrics describe how well the model finds churners, but they answer different questions. Churn is imbalanced (~27% positive), so **accuracy alone is misleading** — a model that predicts "no one churns" would score 73% accuracy while being useless.

### ROC-AUC — *how well does the model rank customers by risk?*
The area under the ROC curve is the probability that a randomly chosen churner is scored higher-risk than a randomly chosen non-churner. It ranges 0.5 (random) → 1.0 (perfect) and is **threshold-independent** — it judges the quality of the risk *scores*, not yes/no decisions. Best single metric for choosing between models when you plan to rank customers and act on the top slice. Our best: **0.843**.

### Recall — *of the customers who actually churned, how many did we catch?*
`Recall = True Positives / (True Positives + False Negatives)`. High recall means few churners slip through undetected. In retention this is often the priority: **a missed churner is a lost customer**, usually costlier than a wasted retention offer. Gradient Boosting catches 51% at the default threshold; the linear/boosting models catch ~78%.

### F1 — *the balance between catching churners and not crying wolf.*
`F1 = harmonic mean of Precision and Recall`. Precision asks "of the customers we flagged, how many really churned?" F1 rewards models that are good at *both* — it's the right summary when both false negatives (missed churners) and false positives (wasted offers) carry real cost.

> **Rule of thumb:** use **ROC-AUC to pick and compare models**, then **tune the decision threshold** against **Recall vs. Precision (F1)** to match how many customers your retention team can actually contact.

---

## Selected model & why

**Gradient Boosting** is selected — it has the top ROC-AUC (0.843), meaning it ranks customers by churn risk better than any other candidate, which is exactly what a "score the base and target the riskiest" retention program needs.

It was then optimized with `RandomizedSearchCV` (30 candidates × 5-fold stratified CV, scoring ROC-AUC):

| Metric | Baseline | Tuned | Δ |
|---|---|---|---|
| ROC-AUC | 0.8426 | **0.8459** | +0.0034 |
| Accuracy | 0.8027 | 0.8062 | +0.0035 |
| Precision | 0.6667 | 0.6836 | +0.0170 |

Best params: `learning_rate=0.01`, `n_estimators=300`, `max_depth=4`, `subsample=0.7`. Cross-validated ROC-AUC reached **0.850**.

> **Production note:** LightGBM and XGBoost land within ~0.005 ROC-AUC of the winner and train much faster. If retraining speed or scale matters more than the last decimal of accuracy, LightGBM is an excellent drop-in alternative — the notebook already tunes whichever model wins.

---

## How to deploy

### 1. Persist the trained pipeline
The model *and* its preprocessing are one `sklearn` Pipeline, so a single artifact is self-contained (no separate encoders/scalers to manage). Add this after tuning in the notebook:

```python
import joblib
joblib.dump(search.best_estimator_, "models/churn_model.joblib")
```

### 2. Score new customers
```python
import joblib, pandas as pd

model = joblib.load("models/churn_model.joblib")
new_customers = pd.read_csv("new_customers.csv")   # same raw columns as training
risk = model.predict_proba(new_customers)[:, 1]     # churn probability 0–1
new_customers["churn_risk"] = risk
```
Pass the **raw** columns — the pipeline applies imputation, encoding, and scaling internally.

### 3. Choose a decision threshold (don't default to 0.5)
Pick the cutoff from business economics, not convenience: if a save-offer is cheap relative to a lost customer, **lower the threshold to raise recall** and catch more churners. Tune it on the precision–recall curve against your retention team's contact capacity.

```python
THRESHOLD = 0.35                       # tuned to team capacity / offer cost
outreach = new_customers[risk >= THRESHOLD]
```

### 4. Serve it
- **Batch (recommended to start):** a scheduled job scores the customer base weekly and routes the top-risk decile to retention outreach, sized by each customer's `MonthlyCharges` (revenue at risk).
- **Real-time:** wrap `predict_proba` in a REST endpoint (FastAPI/Flask) for on-demand scoring.

### 5. Monitor & retrain
- Track input **data drift** and **prediction drift**; watch live ROC-AUC / recall as labels arrive.
- **Retrain on a schedule** as customer behavior shifts.
- **A/B test** interventions to confirm the model drives real retention lift — not just accurate predictions.

---

## Running the notebook

This project uses [**uv**](https://docs.astral.sh/uv/) for package management.

```bash
uv sync                                    # install all dependencies into .venv
```

Register the kernel and open the notebook (select the `Python (churn .venv)` kernel):

```bash
.venv/Scripts/python.exe -m ipykernel install --user --name churn-venv --display-name "Python (churn .venv)"
```

Or re-run the whole notebook headlessly:

```bash
.venv/Scripts/python.exe -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=churn-venv \
  customer_churn_analysis.ipynb
```

### Stack
`pandas` · `numpy` · `scikit-learn` · `xgboost` · `lightgbm` · `matplotlib` · `seaborn`

---

## Project structure

```
customer_churn_coding_agents/
├── data/                                      # source CSV
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── local_coding_agent/                        # ← you are here
│   ├── customer_churn_analysis.ipynb          # full executed analysis
│   ├── Instrunctions.md                       # original task brief
│   ├── pyproject.toml                         # uv-managed dependencies
│   ├── requirements.txt
│   └── README.md                              # this file
├── databricks_notebooks/                      # Databricks version of the pipeline (own README.md)
├── Research_guidelines/                       # metric research behind the choices (own README.md)
└── prompts/                                   # original task briefs
```

The notebook reads the CSV from `../data/`, so run it from `local_coding_agent/`.
