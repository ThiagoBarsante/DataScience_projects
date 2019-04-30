
library(tidyverse)
library(xgboost)

df <- read_csv('../data/WA_Retail-SalesMarketing_-ProfitCost.csv')

## remove records without revenue
df <- na.omit(df)

## Columns used for prediction
col_revenue <- c(
  'Year'
  , 'Product line'
  , 'Product type'
  , 'Product'
  , 'Order method type'
  , 'Retailer country'
  , 'Revenue')


df <- df[col_revenue]

df %>% head(5) 

df <- df %>% mutate_if(is.character, as.factor)

options(repr.plot.width = 9, repr.plot.height = 3)
gg1 <- qplot(data=df, x=`Order method type`, y=Revenue, geom='col', fill=Year, main = 'Revenue by Order method' ) +
  scale_y_continuous(name="Revenue", labels = scales::comma) + coord_flip()  +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

gg2 <- qplot(data=df, x=`Retailer country`, y=Revenue, geom='col', fill=Year, main = 'Revenue by Country'  ) +
  scale_y_continuous(name="Revenue", labels = scales::comma) + coord_flip()  +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

gridExtra::grid.arrange(gg1, gg2, nrow = 1)

options(repr.plot.width = 8, repr.plot.height = 2)

## revenue by product line by year
qplot(data=df, x=`Product line`, y=Revenue, geom='col', fill=Year) +
  scale_y_continuous(name="Revenue", labels = scales::comma) + coord_flip()


options(repr.plot.width = 10, repr.plot.height = 4)
qplot(data=df, x=`Product line`, y=Revenue, geom='boxplot', color=`Product line`, facets = ' Year ~ .') +
  scale_y_continuous(name="Revenue", labels = scales::comma)  + ## mediana = 163000
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## revenue by year -> fazer a simulacao/modelo somente com estes dados
options(repr.plot.width = 8, repr.plot.height = 2)
qplot(data=df, x=Year, y=Revenue, geom='col', fill=Year)+
  scale_y_continuous(name="Revenue", labels = scales::comma) + coord_flip()


## define the target and prepare the data to run XGB
target <- 'Revenue'
idx_target <- which(col_revenue == target)

dfxgb <- df %>% mutate_if(is.character, as.factor)
dfxgb <- dfxgb %>% mutate_if(is.factor, as.integer)

set.seed(12345)
idx <- caret::createDataPartition(dfxgb$Revenue, p=0.80, list=FALSE)
train <- dfxgb[idx, ]
test <- dfxgb[-idx, ]

dtrain <- xgb.DMatrix(data = as.matrix(train[, -idx_target]), label= train[[target]])
dtest <- xgb.DMatrix(data = as.matrix(test[, -idx_target]))

## execution of XGB
set.seed(12345)
fit.xgb <- xgboost(data = dtrain, 
                   objective = "reg:linear",
                   booster = "gbtree",
                   print_every_n = 25, ## print every 25
                   nrounds = 350) 


### feature importance
## importance variable of the xgb model
imp_features <- xgb.importance(model = fit.xgb)

print(imp_features)

## Plot Feature importance
xgb.ggplot.importance(imp_features, rel_to_first = TRUE)


## Evaluate the prediction with test data
predict_xgb <- predict(fit.xgb, dtest)
predict_xgb <- ifelse(predict_xgb < 1, 0, predict_xgb)


## R^2 0.9016625 -> 90% -> 6% better than random forest python (84%)
print('---------------------------- R^2 evaluation')
print(MLmetrics::R2_Score(predict_xgb, test$Revenue))


## The Business Requirement request to predict the revenue for 2008 but do not provide data for 2008

## the trick here => we will use the same products sold on 2007 to predict the Revenue for 2008 and evaluate the results
#### the model used have R^2 of 0.9016 so we expect to achieve a confident result

df_2008 <- dfxgb %>% filter(Year==2007)

## update Year to 2008 and setup the Revenue to 0 -> Revenue will be updated with the prediction later
df_2008$Year <- 2008
df_2008$Revenue <- 0

print('-------------------------------------- INITIAL dataset: 2008 YEAR')
print(summary(df_2008))

## gernerate the data to use with XGB model (fit.xgb)
xgb_2008 <- xgb.DMatrix(data = as.matrix(df_2008[, -idx_target]), label= df_2008[[target]])

predict_revenue_2018 <- predict(fit.xgb, xgb_2008)
predict_revenue_2018 <- ifelse(predict_revenue_2018 < 1, 0, predict_revenue_2018)

## update the Revenue predition for year 2008 and compare with other Years
df_2008$Revenue <- predict_revenue_2018

## generate one dataset to compare the revenue form all Years
df_all <- rbind(dfxgb, df_2008)

print('-------------------------------------- PREDICTION: 2008 YEAR')
summary(df_all)

## info: 2008 is the Revenue Prediction
qplot(data=df_all, x=Year, y=Revenue, geom='col', fill=Year, main='Predicted Revenue for 2008')+
  scale_y_continuous(name="Revenue", labels = scales::comma) + coord_flip()


## Revenue by Year
revenue_by_year <- df_all %>% group_by(Year) %>% summarise(Total_Revenue = sum(Revenue))

print(revenue_by_year)

rev_2008_vs_2005 <- round( (revenue_by_year[revenue_by_year$Year==2008, 'Total_Revenue'] / 
                            revenue_by_year[revenue_by_year$Year==2005, 'Total_Revenue']) * 100  , 2)

rev_2008_vs_2007 <- round( (revenue_by_year[revenue_by_year$Year==2008, 'Total_Revenue'] / 
                            revenue_by_year[revenue_by_year$Year==2007, 'Total_Revenue']) * 100  , 2)

print(' -------------- Revenue 2008 vs 2005 ')
colnames(rev_2008_vs_2005) <- 'Percent Revenue_2008_vs_2005'
rev_2008_vs_2005

print(' -------------- Revenue 2008 vs 2007 ')
colnames(rev_2008_vs_2007) <- 'Percent Revenue_2008_vs_2007'
rev_2008_vs_2007

print('----------------  THE END')
