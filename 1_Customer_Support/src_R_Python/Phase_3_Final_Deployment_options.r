
library(tidyverse)

# Load the Dataset - Customer Churn
df <- readr::read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df %>% head(3)


options(repr.plot.width = 8, repr.plot.height = 3)
## Exclude Customer_ID and apply 0 to Total Charges -> First Bill
df[is.na(df$TotalCharges) & df$tenure==0 , ]['TotalCharges'] <- 0
# df$customerID <- NULL

df %>% filter(Contract=='Month-to-month') %>%   
  group_by(Churn, tenure, Contract) %>% 
  summarise( MonthlyCharges=sum(MonthlyCharges), 
             TotalCharges=sum(TotalCharges))  %>% 
  qplot(data=. ,x=tenure, y=TotalCharges, 
        geom='area', alpha=0.5, fill=(Churn),
        main='    Contract type: Month-to-month') 




### JUST TO REMEMBER, THE MACHINE LEARNING MODEL WAS SAVED FOR FUTURE USE IN PHASE 2

## Save the model -> 80% of accuracy
## xgb.save(fit.xgb, '../data/xgb_model_acc_80p.model')

## obs. to load the model later just run the command below
## model_xgb <- xgb.load('./data/xgb_model_acc_80p.model')
