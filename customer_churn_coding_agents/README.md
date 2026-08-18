# Customer Churn with Coding Agents

End-to-end churn prediction on the **Telco Customer Churn** dataset (IBM sample), built twice with AI coding agents: once on the **Databricks lakehouse** (medallion bronze → silver → ML) and once with a **local coding agent** in a single reproducible notebook.

Part of the [DataScience_projects](../README.md) portfolio — see the root README for the full narrative and the comparison against the hand-coded [`Customer_Support/`](../Customer_Support/) baseline.

- **Dataset:** 7,043 customers × 21 columns — [`data/`](./data/)
- **Task:** binary classification, `Churn` = Yes/No
- **Base rate:** 26.54% churn (1,869 churners) — class imbalance **2.77 : 1**

---

## 1. The prompts

Every prompt used to drive the agents is checked in under [`prompts/`](./prompts/), in execution order. They state the *business question and the artifact to produce* — not the code.

| Prompt | Drives |
|---|---|
| [`z01_prompt_databricks_genie_bronze_table.md`](./prompts/z01_prompt_databricks_genie_bronze_table.md) | Read the Unity Catalog volume; create bronze + silver tables; wire them into a pipeline and run it |
| [`z02_EDA__customer_churn__and_machine_learning_proposal.md`](./prompts/z02_EDA__customer_churn__and_machine_learning_proposal.md) | Comprehensive EDA, correlation heatmaps, anomaly detection, RF baseline, ML use-case proposal |
| [`z03_feature_engineering__machine_learning.md`](./prompts/z03_feature_engineering__machine_learning.md) | Silver table + feature engineering; RF baseline with Gini feature importance |
| [`z04_model_training_and_Evaluation.md`](./prompts/z04_model_training_and_Evaluation.md) | Multi-model comparison, hyperparameter tuning, SHAP, business recommendations |
| [`z05_local_coding_agent_full_pipeline.md`](./prompts/z05_local_coding_agent_full_pipeline.md) | The same CRISP-DM brief, handed to a **local** agent instead |
| [`z06_local_coding_agent_readme_request.md`](./prompts/z06_local_coding_agent_readme_request.md) | Turn the notebook result into a README explaining ROC-AUC vs. Recall vs. F1 and how to deploy |

The full annotated agent transcript — including the errors it hit and how it recovered — is in [`databricks_notebooks/README.md`](./databricks_notebooks/README.md).

---

## 2. Loading the dataset on Databricks (bronze and silver)

Medallion architecture in Unity Catalog, built by the agent from prompt `z01`/`z03`:

| Layer | Table | Contents |
|---|---|---|
| **Source** | `vol_demo.dw_raw.customer_churn` | Raw CSV in a UC volume |
| **Bronze** | `crm.customer_info.customer_churn_bronze` | 7,043 rows × 21 columns, **all typed as string** — faithful landing copy, Delta managed |
| **Silver** | `crm.customer_info.customer_churn_silver` | Cast + cleaned + **28 engineered features**, ML-ready |

**Data-quality fixes applied between bronze and silver**

- `TotalCharges` cast from string → numeric (11 blanks, 0.16%, all at `tenure = 0`)
- `Churn` mapped to a binary `Churn_Binary` target
- Duplicate check: all 7,043 `customerID` values unique

**The 5 engineered feature families**

| Family | Examples |
|---|---|
| Tenure | `Tenure_Group`, `Is_Early_Customer` (0–3 m), `Tenure_Years`, `Is_New_Customer` |
| Spending | `Spending_Tier`, `Spend_Rate`, `Is_High_Value`, `Is_Premium_Customer`, `Spending_Stability` |
| Service bundles | `Num_Internet_Services`, `Total_Services`, `Service_Engagement`, `Has_Streaming`, `Has_Security` |
| Risk indicators | `Is_MonthToMonth`, `Is_Electronic_Check`, `Has_Fiber_Optic`, `Risk_Score`, `Is_High_Risk_Combo` |
| Binary encodings | `gender_Binary`, `Partner_Binary`, `Dependents_Binary`, `PhoneService_Binary`, `PaperlessBilling_Binary` |

---

## 3. Notebooks

### Track 2 — Databricks Data Science Agent → [`databricks_notebooks/`](./databricks_notebooks/)

| Notebook | CRISP-DM phase | Output |
|---|---|---|
| [`00_Bronze_Table_Creation.ipynb`](./databricks_notebooks/00_Bronze_Table_Creation.ipynb) | Data Understanding | Bronze Delta table, 7,043 records |
| [`01_EDA_Baseline_Model_feature_importance.ipynb`](./databricks_notebooks/01_EDA_Baseline_Model_feature_importance.ipynb) | Data Understanding → Modeling | Missing values, distributions, churn cross-tabs, correlation heatmaps, outliers & anomalies — then a raw-feature RF baseline + Gini importance and the recommended ML use cases |
| [`02_silver_table_FE_and_ML_baseline.ipynb`](./databricks_notebooks/02_silver_table_FE_and_ML_baseline.ipynb) | Data Preparation | Silver table + RF baseline (ROC-AUC 0.847, recall 78.3%) |
| [`03_machine_learning_and_model_evaluation.ipynb`](./databricks_notebooks/03_machine_learning_and_model_evaluation.ipynb) | Modeling → Evaluation | 5-model comparison, tuning, SHAP, business recommendations, MLflow tracking |

**Key EDA findings**

| Signal | Finding |
|---|---|
| Onboarding crisis | **56.2% churn in the first 3 months** (1,062 customers) — 2× the base rate |
| Contract type | Month-to-month: **42.7% churn** (3,875 customers) |
| Service quality | Fiber optic: **41.9% churn** (3,096 customers) |
| Payment friction | Electronic check: **45.3% churn** (2,365 customers) |
| Tenure | Churners average 18.0 months vs. 37.6 retained (−52%); tenure↔churn correlation **−0.35** |
| Price | Churners pay **21.5% more** monthly ($74.44 vs. $61.27) |
| Revenue at risk | **$1.67M ARR** (~$139K/month); average customer lifetime value $2,280 |

**Model comparison** (held-out 20%, stratified, silver features)

| Model | ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.8642** | 0.4638 | 0.7033 | 0.5590 | 0.8062 |
| Gradient Boosting | 0.8636 | 0.5684 | 0.6883 | 0.6226 | 0.8176 |
| LightGBM | 0.8557 | 0.5550 | 0.6765 | 0.6097 | 0.8119 |
| Random Forest | 0.8491 | 0.4853 | 0.6679 | 0.5621 | 0.7999 |
| XGBoost | 0.8438 | 0.5174 | 0.6370 | 0.5710 | 0.7942 |

Tuning (`RandomizedSearchCV`, 5-fold, scoring ROC-AUC → `C=1`, `penalty=l2`, `solver=liblinear`) left ROC-AUC flat at **0.8637** while lifting recall 0.464 → **0.526** and F1 0.559 → **0.592**. It moved the operating point; it did not make the model smarter.

### Track 3 — Local coding agent → [`local_coding_agent/`](./local_coding_agent/)

[`customer_churn_analysis.ipynb`](./local_coding_agent/customer_churn_analysis.ipynb) — the same CRISP-DM cycle in one executed notebook, no cloud dependency. Detailed write-up, including the deployment recipe: [`local_coding_agent/README.md`](./local_coding_agent/README.md).

| Model | ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| **Gradient Boosting** ✅ | **0.843** | 0.51 | 0.67 | 0.58 | 0.80 |
| Logistic Regression | 0.842 | 0.78 | 0.50 | 0.61 | 0.74 |
| XGBoost | 0.841 | 0.78 | 0.53 | 0.63 | 0.75 |
| LightGBM | 0.839 | 0.77 | 0.52 | 0.62 | 0.75 |
| Random Forest | 0.822 | 0.64 | 0.55 | 0.59 | 0.77 |

Run it with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

---

## 4. Choosing the metric

Background reading in [`Research_guidelines/`](./Research_guidelines/) — the same brief ("explain classification metrics under imbalance for telecom churn and financial fraud") answered independently by two research agents:

| Research agent | File | Angle |
|---|---|---|
| **ChatGPT** | [`deep-research-report_gptweb.md`](./Research_guidelines/deep-research-report_gptweb.md) | Precision/Recall/F1/AUC-PR under rare events |
| **Claude Opus** | [`research_claude_opus.md`](./Research_guidelines/research_claude_opus.md) | Layered walkthrough from the confusion matrix up |

The short version, for this dataset:

- **Accuracy is the wrong headline.** "Nobody churns" scores 73.5% and saves zero customers.
- **ROC-AUC selects the model** — it measures ranking quality and is threshold-independent.
- **Recall sizes the campaign** — a missed churner is a lost customer; a false alarm is a discount coupon.
- **The threshold is a business decision.** Set it on the precision–recall curve against the retention team's contact capacity, not at the library default of 0.5.

At 26.5% positives, ROC-AUC is still trustworthy here. Push to fraud-scale imbalance (0.1%) and it stops being — that is when PR-AUC and calibration take over.

---

## 5. Final conclusion

**Both agents hit the same ceiling.** Two independent toolchains, two feature sets, five algorithms each, hyperparameter tuning on top — everything lands at **ROC-AUC ≈ 0.84–0.86**. The limit is in the data, not the modelling effort. Recognising that and stopping is a human judgement the agent will not make for you.

**The agents were fast and genuinely useful.** They wrote the medallion pipeline, the EDA, the plots, the encoders, the model comparison and the SHAP analysis from six natural-language prompts — and they self-corrected real bugs mid-run (`.cache()` unsupported on serverless compute, a string/numeric comparison in the missing-value scan, categorical columns leaking into the feature matrix).

**They were also honest about a negative result.** Tuning did not beat the baseline on ROC-AUC, and the notebook says so instead of manufacturing a win.

**And they made mistakes a reviewer had to catch:**

- `customerID` was one-hot encoded in notebook `03`, inflating the feature matrix to **7,105 columns** of near-pure noise. The identifier should have been dropped before encoding.
- `Risk_Score` tops the silver-table feature importance (15.6%) **partly by construction** — it is a weighted composite of the month-to-month, electronic-check and fiber-optic flags, so it restates the EDA rather than discovering anything. The raw-feature ranking in `01_EDA_Baseline_Model_feature_importance` is the more interpretable one: tenure 0.233, MonthlyCharges 0.159, Contract 0.142, OnlineSecurity 0.080, TechSupport 0.064.
- The run logged **"classes are balanced"** at 2.77:1 and skipped `scale_pos_weight`. Defensible at this ratio — but it should be a stated decision, not a silent branch.
- Splits are **random, not out-of-time**. A production churn model needs a temporal holdout.

**The business answer does not depend on the model.** Every track points at the same three levers — the 0–3 month onboarding failure (56% churn), month-to-month contracts (42.7%, 3,875 customers), and fiber-optic service quality (41.9%). A retention programme could act on those today with no model at all; the model exists to *prioritise* the outreach, sized by revenue at risk.

> **Tooling:** Databricks Data Science Agent · Databricks Genie · Unity Catalog · MLflow · Claude Code · `uv` · pandas · scikit-learn · XGBoost · LightGBM · SHAP
