crm.customer_info

vol_demo.dw_raw.customer_churn

** prompt 1 - Databricks **
read file at volume vol_demo.dw_raw.customer_churn and create a bronze and silver tables at crm.customers_info (named... customer_churn_bronze and customer_churn_silver)
Keep all columns as string for bronze table and cast columns for silver.

save all this code and comment in a new notebook at workspace dir already created 
/Workspace/Users/<USER_EMAIL>/Customer_churn/

named 00_customer_churn_bronze and 01_customer_silver for notebooks

** create a pipeline with these notebooks and run it, test if transformations were executed and fix any error if it happen **
