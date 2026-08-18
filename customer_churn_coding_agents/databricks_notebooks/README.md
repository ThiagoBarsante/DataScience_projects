# Databricks notebooks — Track 2 on the lakehouse

Guide to the four notebooks in this directory, followed by the **Databricks Data Science Agent** transcript — kept verbatim and unedited, errors included, because in an agent case study how the work happened is the point.

**Scope:** bronze table → EDA → silver table + baseline model → machine learning and model evaluation.

Each `###` below marks one prompt → response turn; the prompts that drove them are in [`../prompts/`](../prompts/).

| Stage | Notebook | What the agent delivered | In this transcript |
|---|---|---|---|
| Bronze | [`00_Bronze_Table_Creation.ipynb`](./00_Bronze_Table_Creation.ipynb) | `crm.customer_info.customer_churn_bronze` — 7,043 records × 21 columns, 26.54% churn rate | ✅ |
| EDA + baseline | [`01_EDA_Baseline_Model_feature_importance.ipynb`](./01_EDA_Baseline_Model_feature_importance.ipynb) | Missing values, type mismatches, churn cross-tabs, correlations, anomaly detection, RF feature importance | ✅ |
| Silver + ML baseline | [`02_silver_table_FE_and_ML_baseline.ipynb`](./02_silver_table_FE_and_ML_baseline.ipynb) | `customer_churn_silver` — 28+ engineered features in 5 families; Random Forest at **ROC-AUC 0.8466, recall 78.34%** | ✅ |
| Modeling + evaluation | [`03_machine_learning_and_model_evaluation.ipynb`](./03_machine_learning_and_model_evaluation.ipynb) | Five-algorithm comparison, hyperparameter tuning, SHAP explanations, business recommendations | — see the notebook |

**Headline findings from the run:**

- **Onboarding crisis** — churn concentrates in the first months of tenure; early customers are the highest-risk group.
- **Contract and payment terms dominate** (40.9% of predictive power), followed by tenure (17.1%) and services (16.5%).
- **Demographics are near-irrelevant** (~1.4%) — gender, senior status and family make almost no difference.
- **Data quality issues surfaced early**: `TotalCharges` stored as string, plus billing-total mismatches.

> One caveat worth carrying into any reading of this transcript: the agent's top feature, `Risk_Score` (15.56%), is a composite it engineered from the month-to-month, electronic-check and fiber-optic flags — so it partly restates the EDA rather than discovering something new.

---

###

I'll help you read the data from the volume and create a bronze table. Let me start by exploring the volume contents and then create the table.

*Used Execute Code, Check Permissions*

Successfully created the bronze table crm.customer_info.customer_churn_bronze! 

The table contains **7,043 customer records** from the telco customer churn CSV file in the volume. The data includes 21 columns covering customer demographics, service subscriptions, billing information, and churn status. Key highlights:

* **Total customers**: 7,043
* **Churned customers**: 1,869 (26.54% churn rate)
* **Format**: Delta table (managed)
* **Schema**: Includes customerID, demographics (gender, SeniorCitizen), services (PhoneService, InternetService, etc.), billing details (MonthlyCharges, TotalCharges), and Churn indicator

The bronze layer table is now ready for further data transformation and analysis.

###

I'll create a comprehensive EDA notebook for you that analyzes the customer churn bronze table with detailed statistical analysis, visualizations, and data quality insights.

*Used Create Asset, Search Assets, Read Asset By Id, Edit Asset, Open Asset*

Perfect! I've created a comprehensive EDA notebook and opened it for execution. The EDA_v2 notebook includes:

## Analysis Structure:

**1. Data Overview & Configuration**
   - Source table setup and library imports

**2. Missing Values Analysis**
   - Comprehensive null/empty value detection across all columns
   - Percentage calculations and visualizations

**3. Data Type Assessment**
   - Schema inspection
   - Identification of type mismatches (e.g., TotalCharges as string)

**4. Numerical Features Analysis**
   - Statistical summaries (mean, std, quartiles) for tenure, MonthlyCharges, SeniorCitizen
   - Distribution insights and outlier detection

**5. Categorical Features Analysis**
   - Value counts and percentages for all 16 categorical variables
   - Cardinality assessment

**6. Target Variable Deep Dive**
   - Churn distribution and class balance
   - Imbalance ratio calculation

**7. Churn Rates by Key Features**
   - Cross-tabulation analysis showing churn rates across demographics, services, and contract types
   - High-risk segment identification

**8. Correlation Analysis**
   - Numerical feature correlations
   - Tenure vs. churn relationship

**9. Data Quality Assessment**
   - Duplicate detection
   - Logical consistency checks
   - Validation rules

**10. Actionable Insights & Recommendations**
   - Key findings summary
   - Feature engineering suggestions
   - Modeling considerations
   - Business action items

The notebook is now running the complete analysis on your bronze table!

I'll run all the cells in the EDA notebook to perform the comprehensive analysis on the customer churn data.

*Used Run Notebook Cells*

I need to fix the error - `.cache()` is not supported on serverless compute. Let me remove that line and rerun the cells.

*Used Edit Asset, Run Notebook Cells*

I need to fix the missing values analysis - it's trying to compare empty strings with numeric columns. Let me update the logic to handle different data types properly.

*Used Edit Asset, Run Notebook Cells*

Excellent! All cells executed successfully. Let me retrieve the key findings from the analysis.

*Used Fetch Omitted Content*

Perfect! The comprehensive EDA has been completed successfully. Here's a summary of the key findings:

## 📊 Analysis Summary

**Dataset Overview:**
* **7,043 customers** across **21 features**
* All customerIDs are unique (no duplicates)
* Overall data quality is excellent

**Missing Values:**
* Only **11 missing values** (0.16%) in TotalCharges column
* All other columns are complete

**Churn Analysis:**
* **Churn Rate: 26.54%** (1,869 churned / 5,174 retained)
* **Class imbalance ratio: 2.77:1** (slight imbalance)
* Churned customers have **19.6 months LESS tenure** on average (18.0 vs 37.6 months)

**High-Risk Segments:**
* **55%** have month-to-month contracts
* **Fiber optic users: 41.9% churn rate** (significantly higher than average)

**Data Type Issue:**
* TotalCharges is stored as string but should be numeric (needs conversion for modeling)

## 💡 Key Recommendations

**1. Data Quality:**
* Convert TotalCharges to numeric type
* Handle the 11 missing TotalCharges records

**2. Feature Engineering:**
* Create tenure buckets (0-12, 13-24, 25-48, 49+ months)
* Calculate spend rate (TotalCharges/tenure)
* Engineer service bundle features

**3. Modeling:**
* Address class imbalance with SMOTE or class weights
* Focus on early-tenure customers (highest risk)
* Consider separate models for contract types

**4. Business Actions:**
* Target month-to-month customers for loyalty programs
* Investigate fiber optic service quality
* Offer retention incentives at tenure milestones

All analysis results are now available in the EDA_v2 notebook. The notebook is ready for the next phase: feature engineering and model development.

#####################################


I'll enhance the EDA notebook by adding correlation heatmaps and comprehensive outlier/anomaly analysis. Let me add new cells with visualizations and deeper insights.

*Used Edit Asset*

Now let me run these new analysis cells to generate the correlation heatmaps, outlier detection, anomaly analysis, and comprehensive business insights.

*Used Run Notebook Cells*

Perfect! All analysis cells executed successfully. Let me retrieve the key findings from the new analysis.

*Used Fetch Omitted Content*

Excellent! I've completed comprehensive correlation analysis, outlier detection, anomaly identification, and business insights analysis. Here are the critical findings:

## 📊 **Correlation Analysis & Heatmap**

**Key Correlations:**
* **Tenure ↔ Churn: -0.35** (Moderate negative) - **Most important predictor!**
  - Higher tenure = Lower churn (protective factor)
  - Churned customers have **52.1% lower tenure** (18 vs 37.6 months)
* **MonthlyCharges ↔ Churn: +0.19** (Weak positive)
  - Churned customers pay **21.5% MORE** monthly ($74.44 vs $61.27)
* **Tenure ↔ MonthlyCharges: +0.25** (Weak)
  - Largely independent - loyalty doesn't correlate with pricing tier

## 🚨 **Critical Anomalies Detected**

### 1. **ONBOARDING CRISIS**
* **56.21% churn rate** in first 3 months (1,062 customers)
* **2x higher than average** - indicates serious onboarding problems
* **Action Required:** Immediate review of first 90-day customer experience

### 2. **Billing Calculation Errors**
* **2,602 customers (37%)** have TotalCharges mismatches
* Expected vs actual billing discrepancies (>$50 or >10%)
* Lower churn rate among these customers (19.52%) - may indicate billing system issues rather than fraud

### 3. **High-Value Customer Loss**
* **393 premium customers lost** (>$80/month, >2 years tenure)
* **$462,102 annual revenue loss** from this segment alone
* All churned with month-to-month contracts + fiber optic service

## 💰 **Business Impact**

**Revenue at Risk:**
* **$1,669,570 annual recurring revenue** lost to churn
* **$139,131 monthly revenue** walking out the door
* Average customer lifetime value: **$2,280**

**High-Risk Segments:**
1. **Month-to-month contracts: 42.71% churn** (3,875 customers)
2. **Fiber optic users: 41.89% churn** (3,096 customers)
3. **Electronic check payment: 45.29% churn** (2,365 customers)
4. **High-value customers (≥$70/mo): 35.48% churn** (3,591 customers)

## 🎯 **No Statistical Outliers BUT Major Extreme Values**

Using IQR method:
* **No traditional outliers** in tenure or monthly charges
* However, extreme segments identified:
  * **15% of customers** have ≤3 months tenure (critical period)
  * **13% pay >$100/month** (all fiber optic, 28% churn rate)
  * **11 zero-tenure customers** (brand new, at risk)

## 🔥 **Immediate Action Items**

**PRIORITY 1 - IMMEDIATE:**
1. **Fix onboarding process** - 56% churn in first 3 months is catastrophic
2. **Investigate fiber optic service quality** - 42% churn rate suggests systemic issues
3. **Target month-to-month customers** for contract upgrades - could save $700K+ annually

**PRIORITY 2 - HIGH:**
4. **Early warning system** for customers <12 months tenure
5. **Retention program** for high-value customers (>$80/month)
6. **Review electronic check** payment experience (45% churn rate)

**PRIORITY 3 - MEDIUM:**
7. **Loyalty rewards** at tenure milestones (6, 12, 24 months)
8. **Audit billing system** for the 2,602 calculation mismatches

The correlation heatmap and full analysis details are now available in your EDA_v2 notebook with 5 new advanced analysis cells!

#####################################

#####################################
I'll create a comprehensive notebook that builds a silver table with feature engineering, trains a baseline Random Forest model, and analyzes feature importance. Let me start by reading the existing EDA results, then create the new notebook.

*Used Read Asset By Id, Create Asset, Edit Asset*

Perfect! I've created a comprehensive notebook **03_silver_table_FE_and_ML_baseline** with 23 cells. Let me open it so you can run the analysis.

*Used Open Asset*

Excellent! I've created and opened the comprehensive **03_silver_table_FE_and_ML_baseline** notebook. Here's what it includes:

## 📋 **Notebook Structure (23 Cells)**

### **Part 1: Silver Table Creation & Feature Engineering**

**Data Quality Fixes:**
* Convert TotalCharges from string → numeric
* Handle missing values for tenure=0 customers
* Create binary churn target (0/1)

**5 Feature Engineering Categories (28+ new features):**

1. **Tenure Features** (4 features)
   - `Tenure_Group`: Risk-based segments (Very_New_0-3m, New_4-12m, etc.)
   - `Is_Early_Customer`: Critical first 3 months flag
   - `Tenure_Years`: Continuous measure
   - `Is_New_Customer`: Brand new flag

2. **Spending Features** (5 features)
   - `Spending_Tier`: Low/Medium/High segments
   - `Spend_Rate`: Average monthly over tenure
   - `Is_High_Value`: ≥$70/month flag
   - `Is_Premium_Customer`: High value + long tenure
   - `Spending_Stability`: Spend consistency measure

3. **Service Bundle Features** (7 features)
   - `Num_Internet_Services`: Count of add-ons (0-6)
   - `Total_Services`: Overall service count
   - `Service_Engagement`: Categorized engagement
   - `Has_Streaming`, `Has_Security`, etc.

4. **Risk Indicators** (7 features)
   - `Is_MonthToMonth`: Highest churn contract
   - `Is_Electronic_Check`: High-risk payment
   - `Has_Fiber_Optic`: Service quality issue segment
   - `Risk_Score`: Weighted composite score
   - `Is_High_Risk_Combo`: Multiple risk factors

5. **Binary Encodings** (5 features)
   - Convert Yes/No → 0/1 for ML readiness

### **Part 2: Baseline Machine Learning Model**

**Random Forest Configuration:**
* 100 trees, max_depth=10
* Class-weighted (handles 2.77:1 imbalance)
* Train/test split: 80/20 stratified

**Feature Importance Analysis:**
* **Gini Index** rankings (Mean Decrease Impurity)
* Top 20 features visualization
* Cumulative importance curve
* Category-level aggregation

**Visualizations Included:**
1. **Confusion Matrix** with percentages
2. **ROC Curve** with AUC score
3. **Feature Importance Bar Chart** (top 20)
4. **Cumulative Importance Plot**
5. **Category-Level Importance** breakdown

**Performance Metrics:**
* Accuracy, Precision, Recall, F1-Score
* ROC-AUC score
* Detailed classification report
* Feature importance rankings

## 🎯 **What You'll Discover**

The analysis will identify:
* **Top 5 most influential features** driving churn
* **Feature importance percentages** (Gini Index)
* Which **feature categories** matter most (Tenure vs Spending vs Services vs Risk)
* **Model performance baseline** for comparison
* **Actionable insights** for retention strategies

## 💾 **Output Artifacts**

1. **Silver Table**: `crm.customer_info.customer_churn_silver`
   - ML-ready dataset with all engineered features
   - Can be used for future model iterations

2. **Feature Importance Rankings**
   - Identify which features to focus on
   - Guide future feature engineering
   - Support business decision-making

3. **Baseline Model Performance**
   - Benchmark for advanced models
   - ROC-AUC and confusion matrix metrics

The notebook is now ready to execute - run all cells to complete the full pipeline from bronze → silver → baseline ML model with feature importance analysis!

I'll execute all cells in the notebook to build the silver table, train the Random Forest model, and generate comprehensive feature importance analysis. Let me first check the remaining cells, then run everything.

*Used Notebook Read*

Perfect! Now let me execute all the cells to build the silver table, train the model, and generate comprehensive feature importance analysis. I'll run all Python cells in sequence.

*Used Run Notebook Cells*

Excellent! Silver table created successfully with all engineered features. Now let me run the ML model training and evaluation cells.

*Used Run Notebook Cells*

I'll read the remaining cells to see what's in the final section of the notebook and confirm the completion status.

*Used Notebook Read*

Let me check the execution results for the cells that should have been run:

*Used Notebook Get Cells Results*

I can see that cells 18-22, which contain the ROC curve, feature importance analysis, and key insights, have **not been executed yet**. These are critical for completing your request. Let me run them now to generate the feature importance rankings, ROC curve, and identify the most influential churn drivers.

*Used Run Notebook Cells*

The variables from the model training cells aren't in memory. Let me check if the earlier cells were actually executed:

*Used Notebook Get Cells Results*

I can see the issue - the model training cell (15) started but didn't complete, and the evaluation cell (16) was never run. That's why the variables aren't available. Let me run the complete sequence from model training through the final insights:

*Used Run Notebook Cells*

There's a data type issue - some categorical string columns weren't properly excluded from the feature matrix. Let me fix cell 13 to exclude the original Yes/No columns since we created binary versions:

*Used Edit Asset*

Now let me re-run the cells from feature selection onward to complete the ML pipeline:

*Used Run Notebook Cells*

Excellent! The model training and evaluation are complete. Now let me run the remaining cells for ROC curve, feature importance analysis, and insights:

*Used Run Notebook Cells*

Excellent! All cells have executed successfully. Let me get the detailed results from the key analysis cells:

*Used Fetch Omitted Content, Notebook Get Cells Results*

Perfect! All 5 objectives have been successfully completed. Here's what was accomplished:

## ✅ Pipeline Execution Complete

### 1. **Silver Table Creation**
* Built crm.customer_info.customer_churn_silver with **28+ engineered features** based on EDA insights
* Successfully transformed 7,043 records across 5 feature categories:
  * **Tenure features**: Early customer flags, tenure buckets, years
  * **Spending features**: Value tiers, spend rates, premium indicators
  * **Service features**: Bundle counts, engagement levels
  * **Risk indicators**: Compound risk scores, high-risk combinations
  * **Binary encodings**: Yes/No conversions

### 2. **Baseline Model Performance**
Random Forest classifier achieved strong results:
* **Accuracy**: 75.87% - solid overall performance
* **Recall**: 78.34% - captures most churners (critical for retention)
* **ROC-AUC**: 0.8466 - excellent discrimination capability
* **Precision**: 53.08% - indicates room for refinement

### 3. **Feature Importance Rankings (Gini Index)**

**Top 5 Most Influential Features:**
1. **Risk_Score** (15.56%) - Compound risk indicator
2. **Is_High_Risk_Combo** (9.86%) - Multiple risk factors present
3. **Is_MonthToMonth** (9.77%) - Contract type flag
4. **tenure** (6.80%) - Customer lifetime
5. **Tenure_Years** (5.62%) - Tenure in years

**Importance Concentration:**
* Top 5 features: 47.62% of predictive power
* Top 10 features: 67.53% of predictive power
* Top 20 features: 85.34% of predictive power

### 4. **Confusion Matrix & ROC Curves**
* Confusion matrix visualization created showing true/false positives/negatives
* ROC curve plotted with AUC = 0.8466 (excellent discrimination)

### 5. **Most Influential Churn Drivers**

**By Category:**
* **Contract/Payment factors** (40.9%) - Dominant category
  * Month-to-month contracts
  * Electronic check payments
  * Paperless billing
* **Tenure factors** (17.1%) - Second most important
  * Early customers (0-3 months) at highest risk
  * Long tenure is protective
* **Service factors** (16.5%) - Third priority
  * Fiber optic service quality issues
  * Number of add-on services
* **Spending patterns** (14.1%) - Fourth driver

**Key Finding**: Demographics (gender, senior status, family) have minimal impact (1.4%), while **contract terms and risk indicators dominate churn prediction**.

View confusion matrix View ROC curve View feature importance