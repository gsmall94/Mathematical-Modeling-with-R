# Geoffrey Small
# Math 456 Project, Part 1
# 02/27/2024

# Note: data from https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset

# Description: this code presents the case where the data are first cleaned by removing outliers via Box-Whisker plots



#.....................................................................................................
        ## Preliminary Steps ##

# Import packages:
library(ggplot2)    # data visualization, plotting
library(dplyr)
library(tidyverse)  
library(EnvStats)   # for QQPlot() 
library(ISLR)       # for Cook's distance method
library(MLmetrics)  # for RMSE() function

# Set seed for sampling during partitioning data set:
set.seed(1234)


# Set working directory and import csv data set:
setwd("/Users/geoffreysmall/Documents/Math456_Project/Part1")
dt <- read.csv('/Users/geoffreysmall/Documents/Math456_Project/Part1/insurancedata.csv', header = T)



#.....................................................................................................
        ## Initial data assessment (visualization, cleaning, etc.) ##

plot(dt$bmi, dt$charges)  #visualize data of charges vs bmi. do we see outliers?
title(main = list("Charge Amount vs. BMI (Before Cleaning)", font = 1 ))
plot(dt$age, dt$charges)  #visualize data of charges vs age do we see outliers?
title(main = list("Charge Amount vs. Age (Before Cleaning)", font = 1))

which(is.na(dt))  # check for NA's
sum(is.na(dt))    # How many NA's are found?

summary(dt)      # data set summary info
boxplot(dt$bmi)  #outliers observed
title(main = list("BMI", font = 1))
boxplot(dt$age)  #no outliers observed
title(main = list("Age", font = 1))
boxplot(dt$charges) #outliers observed
title(main = list("Charges", font = 1))

# Remove outliers (via Box-Whisker plot) and revisualize:
Q <- quantile(dt$bmi, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(dt$bmi)
up <-  Q[2]+1.5*iqr # Upper Range  
low<- Q[1]-1.5*iqr # Lower Range

dt_cleanedBMI1 <- subset(dt, dt$bmi > (Q[1] - 1.5*iqr) & dt$bmi < (Q[2]+1.5*iqr))                   
boxplot(dt_cleanedBMI1$bmi)
title(main = list("BMI (Cleaned)", font = 1))

# Remove response (charges) outliers from BMI set:
Q <- quantile(dt_cleanedBMI1$charges, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(dt_cleanedBMI1$charges)
up <-  Q[2]+1.5*iqr # Upper Range  
low<- Q[1]-1.5*iqr # Lower Range

dt_cleanedBMI2 <- subset(dt_cleanedBMI1, dt_cleanedBMI1$charges > (Q[1] - 1.5*iqr) & dt_cleanedBMI1$charges < (Q[2]+1.5*iqr))                   
boxplot(dt_cleanedBMI2$charges)
title(main = list("Charges (Cleaned, BMI as pred)", font = 1))

# But we must also remove the outlier data points corresponding to Age:

# Cleaning data w.r.t. Age:
Q <- quantile(dt$age, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(dt$age)
up <-  Q[2]+1.5*iqr # Upper Range  
low<- Q[1]-1.5*iqr # Lower Range

dt_cleanedAge1 <- subset(dt, dt$age > (Q[1] - 1.5*iqr) & dt$age < (Q[2]+1.5*iqr))                   
boxplot(dt_cleanedAge1$bmi)

# Remove response (charges) outliers from Age set:
Q <- quantile(dt_cleanedAge1$charges, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(dt_cleanedAge1$charges)
up <-  Q[2]+1.5*iqr # Upper Range  
low<- Q[1]-1.5*iqr # Lower Range

dt_cleanedAge2 <- subset(dt_cleanedAge1, dt_cleanedAge1$charges > (Q[1] - 1.5*iqr) & dt_cleanedAge1$charges < (Q[2]+1.5*iqr))                   
boxplot(dt_cleanedAge2$charges)
title(main = list("Charges (Cleaned, Age as pred)", font = 1))

# Now, we have created two data sets, as one simple lin reg will use BMI as predictor and other will use Age:
# These data sets are: dt_cleanedBMI2 and dt_cleanedAge2
write.csv(dt_cleanedBMI2, "/Users/geoffreysmall/Documents/Math456_Project/Part1/cleanedbyBMI.csv", row.names=FALSE)

#.....................................................................................................
        ## Normalcy assessment (Q-Q plot):

# i. General Q-Q plot (quantiles of one numeric variable against the quantiles of a second numeric variable)
# For BMI vs charges variables:
x1 <-(dt_cleanedBMI2$bmi)
y1 <- (dt_cleanedBMI2$charges)
qqplot(x1,y1, xlab="BMI", ylab="Charge Amount", main="Q-Q Plot (BMI Model)")  #Deviation from normalcy seen


# For age vs charges variables:
x2 <-(dt_cleanedAge2$age)
y2 <- (dt_cleanedAge2$charges)
qqplot(x2,y2, xlab="Age", ylab="Charge Amount", main="Q-Q Plot (Age Model)")  #Deviation from normalcy seen


# ii. Normal Q-Q plot (plotting the quantiles of a numeric variable against the quantiles of a normal distribution)

# For BMI model:
qqPlot(dt_cleanedBMI2$charges) # Normal QQ plot for response var
qqPlot(dt_cleanedBMI2$bmi)     # Normal QQ plot for BMI predictor var

# For Age model:
qqPlot(dt_cleanedAge2$charges) # Normal QQ plot for response var
qqPlot(dt_cleanedAge2$age)     # Normal QQ plot for Age predictor var



#......................................................................................................
        ## Splitting Data 50/50 into Training and Test Sets ##
 
# Partition data into 50% test set:
splitPercentage_BMI <- round(nrow(dt_cleanedBMI2) %*% 0.5)  # Split 50% of cleaned BMI data set
splitPercentage_Age <- round(nrow(dt_cleanedAge2) %*% 0.5)  # Split 50% of cleaned Age data set

idxBMI <- sample(1:nrow(dt_cleanedBMI2), splitPercentage_BMI)  # BMI, sample of indices to use
idxAge <- sample(1:nrow(dt_cleanedAge2), splitPercentage_Age)  # Age, sample of indices to use

trainingSet_BMI <- dt_cleanedBMI2[idxBMI, ]   # BMI training set
trainingSet_Age <- dt_cleanedAge2[idxAge, ]   # Age training set

testSet_BMI <- dt_cleanedBMI2[-idxBMI, ]   # BMI test set
testSet_Age <- dt_cleanedAge2[-idxAge, ]   # Age test set


#......................................................................................................
        ## Creating the Models ##

# Model 1: Fit a simple linear regression model (charge vs bmi ) to training data:
fit1 <- lm(charges ~ bmi, trainingSet_BMI)   # BMI is predictor
fit1

# Model 2: Fit a simple linear regression model (charge vs age ) to training data:
fit2 <- lm(charges ~ age, trainingSet_Age)   # Age is predictor
fit2


# Training set predictions:
preds1 <- predict(fit1, trainingSet_BMI)   # BMI
preds2 <- predict(fit2, trainingSet_Age)   # Age


# Get RMSE for Training Data:
RMSE(y_pred= preds1, y_true= trainingSet_BMI$charges)   # training set, model with BMI
RMSE(y_pred= preds2, y_true= trainingSet_Age$charges)   # training set, model with Age



#......................................................................................................
        ## Apply Model to Test Data ##

# Apply to test set:
pred1_Test      <- predict(fit1, testSet_BMI)   # BMI
pred2_Test      <- predict(fit2, testSet_Age)   # Age

# RMSE for the two models:
RMSE(y_pred = pred1_Test, y_true = testSet_BMI$charges)   # test set, model with BMI
RMSE(y_pred = pred2_Test, y_true = testSet_Age$charges)   # test set, model with Age



#......................................................................................................
        ## Additional Model Performance Evaluation ##

# Summary data on models (residuals, R^2, p-vals, etc.):
summary(fit1)  # BMI as predictor
summary(fit2)  # Age as predictor


# Visualize fit of cleaned data sets via ggplot:
# BMI model:
ggplot(dt_cleanedBMI2, aes(bmi, charges)) +
  geom_point() +
  stat_smooth(method = lm)

# Age model:
ggplot(dt_cleanedAge2, aes(age, charges)) +
  geom_point() +
  stat_smooth(method = lm)



