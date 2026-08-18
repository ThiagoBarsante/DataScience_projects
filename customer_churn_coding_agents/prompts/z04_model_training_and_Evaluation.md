
## setup : first change Databricks environment from base environment to ML
### also run pip install xgboost, lightghm and restart kernel

SOURCE_FILE_DATA_DIR = table = crm.customer_info.customer_churn_silver

ALL Notebooks with information are provided at /Workspace/Users/<USER_EMAIL>/Customer_churn

- Train multiple machine learning models on our prepared dataset to predict wine quality using the most important features. Compare different algorithms (e.g., Random Forest, Gradient Boosting, Linear models, xgboost and lightghm) and evaluate them using appropriate metrics. Show me model performance comparisons and feature importance.

- Perform hyperparameter tuning on the best-performing model from our comparison. Use techniques like grid search or random search to optimize model performance. Show me the improvement in metrics after tuning.

Provide also 

# ## Model Interpretation and Business Insights
# 
# The final step involves interpreting your machine learning results and extracting actionable business insights.

# %% [markdown]
# ### Feature Importance and Model Explainability
# 
# ***Analyze the feature importance of our final model. Create visualizations showing which features are most predictive and explain what this means for the business. Use SHAP values or other explainability techniques if appropriate. DO NOT INSTALL SHAP, just try to run ***
# 
# %% [markdown]
# ### Business Recommendations and Next Steps
# 
# ***Based on our EDA and machine learning results, provide business recommendations and actionable insights. What should stakeholders know about the patterns we discovered? What are the next steps for deploying or improving this model?***

