# Geoffrey Small
# Math 456 Project, Part 1 (uncleaned case, with all outliers present)
# 02/27/2024

# Note: data from https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset

# Description: this code presents the case where the data not cleaned. All outliers are present



#.....................................................................................................
        ## Preliminary Steps ##

# Import packages:
library(ggplot2)    # data visualization, plotting
library(dplyr)
library(tidyverse)
library(EnvStats)   # for QQPlot() 
library(MLmetrics)  # for RMSE() function

# Set seed for sampling during partitioning data set:
set.seed(1234)



# Set working directory and import csv data set:
setwd("/Users/geoffreysmall/Documents/Math456_Project/Part1")
dt_unclean <- read.csv('/Users/geoffreysmall/Documents/Math456_Project/Part1/insurancedata.csv', header = T)



#.....................................................................................................
        ## Initial data assessment (visualization, cleaning, etc.) ##

plot(dt_unclean$bmi, dt_unclean$charges)  #visualize data of charges vs bmi. do we see outliers?
plot(dt_unclean$age, dt_unclean$charges)  #visualize data of charges vs age do we see outliers?

which(is.na(dt_unclean))  # check for NA's
sum(is.na(dt_unclean))    # How many NA's are found?

summary(dt_unclean)      # data set summary info

boxplot(dt_unclean$bmi)  #outliers observed
boxplot(dt_unclean$age)  #no outliers observed
boxplot(dt_unclean$charges) #outliers observed


# NO REMOVAL OF OUTLIERS 



#.....................................................................................................
        ## Normalcy assessment (Q-Q plot):

# i. General Q-Q plot (quantiles of one numeric variable against the quantiles of a second numeric variable)
# For BMI vs charges variables:
x1 <-(dt_unclean$bmi)
y1 <- (dt_unclean$charges)
qqplot(x1,y1, xlab="BMI", ylab="Charge Amount", main="Q-Q Plot")  #Deviation from normalcy seen

# For age vs charges variables:
x2 <-(dt_unclean$age)
y2 <- (dt_unclean$charges)
qqplot(x2,y2, xlab="Age", ylab="Charge Amount", main="Q-Q Plot")  #Deviation from normalcy seen


# ii. Normal Q-Q plot (plotting the quantiles of a numeric variable against the quantiles of a normal distribution)
qqPlot(dt_unclean$charges)  # Normal QQ plot for response var
qqPlot(dt_unclean$bmi)      # Normal QQ plot for BMI predictor var
qqPlot(dt_unclean$age)      # Normal QQ plot for Age predictor var



#......................................................................................................
        ## Splitting Data 50/50 into Training and Test Sets ##

# Partition data into 50% test set:
splitPercentage <- round(nrow(dt_unclean) %*% 0.5)
idx <- sample(1:nrow(dt_unclean), splitPercentage)  #sample of indices to use

trainingSet_unclean <- dt_unclean[idx, ]
testSet_unclean <- dt_unclean[-idx, ]



#......................................................................................................
        ## Creating the Models ##

# Model 1: Fit a simple linear regression model (charge vs bmi ) to training data:
fit1_unclean <- lm(charges ~ bmi, trainingSet_unclean)
fit1_unclean

# Model 2: Fit a simple linear regression model (charge vs age ) to training data:
fit2_unclean <- lm(charges ~ age, trainingSet_unclean)
fit2_unclean


# Training set predictions:
preds1_unclean <- predict(fit1_unclean, trainingSet_unclean)   # model with BMI
preds2_unclean <- predict(fit2_unclean, trainingSet_unclean)   # model with Age



# Get RMSE for Training Data:
RMSE(y_pred= preds1_unclean, y_true= trainingSet_unclean$charges)   # model with BMI
RMSE(y_pred= preds2_unclean, y_true= trainingSet_unclean$charges)   # model with Age



#......................................................................................................
        ## Apply Model to Test Data ##

# Apply to test set:
pred1unclean_Test <- predict(fit1_unclean, testSet_unclean)
pred2unclean_Test <- predict(fit2_unclean, testSet_unclean)

# RMSE for the two models:
RMSE(y_pred = pred1unclean_Test, y_true = testSet_unclean$charges)
RMSE(y_pred = pred2unclean_Test, y_true = testSet_unclean$charges)



#......................................................................................................
        ## Additional Model Performance Evaluation ##

# Summary data on models (residuals, R^2, p-vals, etc.)
summary(fit1_unclean)  # BMI as predictor
summary(fit2_unclean)  # Age as predictor


# Visualize fit of uncleaned data sets via ggplot:
# BMI model:
ggplot(dt_unclean, aes(bmi, charges)) +
  geom_point() +
  stat_smooth(method = lm)

# Age model:
ggplot(dt_unclean, aes(age, charges)) +
  geom_point() +
  stat_smooth(method = lm)


