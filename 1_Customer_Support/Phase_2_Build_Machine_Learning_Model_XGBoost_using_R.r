
## Load libraries used in the process
library(tidyverse)
library(caret)
library(xgboost)

## Load the Dataset - Customer Churn
df <- read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
head(df, 5)

## Exclude Customer_ID and apply 0 to Total Charges -> First Bill
df[is.na(df$TotalCharges) & df$tenure==0 , ]['TotalCharges'] <- 0
df$customerID <- NULL

target <- 'Churn'
feature_categories <- df %>% keep(is.character) %>% colnames()
feature_categories <- setdiff(feature_categories, target)

## converte character to factor for analysis
df <- df %>%  mutate_if(is.character, as.factor)

current_features <- c('tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'PaymentMethod' , 'Churn', 'Contract')
summary(df[current_features])


options(repr.plot.width = 8, repr.plot.height = 3)
gg1 <- qplot(data=df, x=Churn, y=MonthlyCharges, color=gender, geom='boxplot') + coord_flip()
gg2 <- qplot(data=df, x=TotalCharges, y=MonthlyCharges, color=gender, geom='point')
# gg3 <- qplot(data=df, x=TotalCharges, y=MonthlyCharges, color=Churn, geom='point')

gridExtra::grid.arrange(gg1, gg2, nrow = 1)

## Payment Method
pmnt <-  df %>% group_by(PaymentMethod, Churn) %>% summarise(customers_n = n())
pmnt$PaymentMethod <-  str_remove(str_remove(str_remove(pmnt$PaymentMethod, 'automatic'), '\\('), '\\)')


## plot de 4 graficos em conjunto
color_manual <- c('#2a39ff', '#ff8b24')

gg3 <- qplot(data=df, x=TotalCharges, y=MonthlyCharges, color=Churn, geom='point')
gg3 <- gg3 + scale_color_manual(values=color_manual)
gg4 <- qplot(data = pmnt, x= PaymentMethod , y=customers_n, fill=Churn,  geom='col') + coord_flip()
gg4 <- gg4 + scale_fill_manual(values=color_manual)


gridExtra::grid.arrange(gg4, gg3, nrow = 1)


## Contract and tenure drives to Churn
gg5 <- qplot(data=df, x=tenure, y=MonthlyCharges, color=Churn, geom='point', facets = .~Contract, main='   Contract Type')
gg5 <- gg5 + scale_color_manual(values=color_manual)

gridExtra::grid.arrange(gg5, nrow = 1)

## few steps before run the xgb model

target <- 'Churn'
feature_categories <- df %>% keep(is.factor) %>% colnames()
feature_categories <- setdiff(feature_categories, target)

## function to run the One Hot Enconding -> all features categories 
ffm_One_Hot_Encoding_dataframe <- function(dataframe, cols_OneHOtEncoding=c()){
  # ONE HOT ENCODING --------------------------------------------------------
  require(caret)  
  
  for (i in seq_along(cols_OneHOtEncoding)){
    idx_col <- which(colnames(dataframe) == cols_OneHOtEncoding[i])
    formula <- as.formula( paste0('~ ' , cols_OneHOtEncoding[i]))    
    dummies <- predict(dummyVars(formula, data = dataframe), newdata = dataframe)
    dataframe <- cbind(dataframe, dummies)
    dataframe[, idx_col] <- NULL
  }
  return(dataframe)  
}

df_xgb <- ffm_One_Hot_Encoding_dataframe(df, feature_categories)

df_xgb$Churn <- as.integer(as.factor(df_xgb$Churn)) -1L

## Dataframe structure
## str(df)

dim(df_xgb)[2]

## Finally let's run the model
## 

set.seed(458)
idx <- createDataPartition(df_xgb$Churn, p=0.80, list = FALSE)
train <- df_xgb[idx, ]
test <- df_xgb[-idx, ]
idx_target <- which(colnames(df_xgb)==target)

dtrain <- xgb.DMatrix(as.matrix(train[, -idx_target]), label=train[[target]])
dtest <-  xgb.DMatrix(as.matrix(test[, -idx_target]), label=test[[target]])


parameters <- list(objective = "binary:logistic", 
                   eval_metric = "auc")

nrounds_xgb <- 100
set.seed(458)
fit.xgb <- xgb.train(params = parameters,
                     data = dtrain,
                     watchlist = list(train = dtrain, eval = dtest),
                     early_stopping_rounds = 5,
                     nrounds = nrounds_xgb,
                     print_every_n = 10,
                     nthread = 2)


xgb_predict <- predict(fit.xgb, as.matrix(test[, -idx_target]))
xgb_predict <- round(xgb_predict)

confusionMatrix(factor(xgb_predict), factor(test$Churn), positive = '1')


## Save the model -> 80% of accuracy
xgb.save(fit.xgb, '../data/xgb_model_acc_80p.model')

## obs. to load the model later just run the command below
## model_xgb <- xgb.load('./data/xgb_model_acc_80p.model')

## importance matrix
importance_matrix <- xgb.importance(colnames(dtrain), 
                                    model = fit.xgb)

options(repr.plot.width = 6, repr.plot.height = 4)
xgb.plot.importance(importance_matrix, top_n = 10)


options(repr.plot.width = 5, repr.plot.height = 3)
gg6 <- qplot(data=df, Contract, fill=Churn, ylab='Number of Contracts') + coord_flip()
gg6 <- gg6 + scale_fill_manual(values=color_manual)

print(gg6)

df %>% filter(Churn=='Yes') %>%   
  group_by(tenure, Contract) %>% 
  summarise( MonthlyCharges=sum(MonthlyCharges), 
             TotalCharges=sum(TotalCharges))  %>% 
  qplot(data=. ,x=tenure, y=MonthlyCharges, 
        geom='area', alpha=0.5, fill=Contract,
        main='    Revenue stopped: Customer Churn = Yes') 
