# Geoffrey Small
# Math 456 Project, Part 1 (using Cook's distance for outliers)
# 02/27/2024

# Note: data from https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset

# Description: this code presents the case where the data are first cleaned by removing outliers via Cook's Distance Method



#.....................................................................................................
        ## Preliminary Steps ##

# Import packages:
library(ggplot2)    # data visualization, plotting
library(dplyr)      # for Cook's distance (glimpse())
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
plot(dt$age, dt$charges)  #visualize data of charges vs age do we see outliers?

which(is.na(dt))  # check for NA's
sum(is.na(dt))    # How many NA's are found?

summary(dt)   # data set summary info



#.....................................................................................................
        ## Normalcy assessment (Q-Q plot):

# i. General Q-Q plot (quantiles of one numeric variable against the quantiles of a second numeric variable)
# For BMI vs charges variables:
x1 <-(dt$bmi)
y1 <- (dt$charges)
qqplot(x1,y1, xlab="BMI", ylab="Charge Amount", main="Q-Q Plot")  #Deviation from normalcy seen

# For age vs charges variables:
x2 <-(dt$age)
y2 <- (dt$charges)
qqplot(x2,y2, xlab="Age", ylab="Charge Amount", main="Q-Q Plot")  #Deviation from normalcy seen


# ii. Normal Q-Q plot (plotting the quantiles of a numeric variable against the quantiles of a normal distribution)
qqPlot(dt$charges) # Normal QQ plot for response var
qqPlot(dt$bmi)     # Normal QQ plot for BMI predictor var
qqPlot(dt$age)     # Normal QQ plot for Age predictor var


#......................................................................................................
        ## Create a prelim model and check for outliers ##

# Reference: https://umamherst.instructure.com/courses/8134/files/4075315?module_item_id=720949

preModel1 <- lm(charges ~ bmi, data = dt)   # BMI
preModel2 <- lm(charges ~ age, data = dt)   # Age

# Diagnostics plots:
par(mfrow = c(2,2))
plot(preModel1)     # BMI
plot(preModel2)     # Age

# Using Cook's Distance to find influential points:
cooksD1 <- cooks.distance(preModel1)   # BMI
cooksD2 <- cooks.distance(preModel2)   # Age

plot(cooksD1,type="b",pch=18,col="red")   # BMI
plot(cooksD2,type="b",pch=18,col="red")   # Age

influential_BMI <- cooksD1[(cooksD1 > (3 * mean(cooksD1, na.rm = TRUE)))]
influential_Age <- cooksD2[(cooksD2 > (3 * mean(cooksD2, na.rm = TRUE)))]

influential_BMI
influential_Age
length(influential_BMI)  #number of influential BMI data pts
length(influential_Age)  #number of influential Age data pts


# Remove these influential points and refit the model:
names_of_influential_BMI <- names(influential_BMI)
names_of_influential_Age <- names(influential_Age)


outliers_BMI <- dt[names_of_influential_BMI,]
outliers_Age <- dt[names_of_influential_Age,]

# We will want the BMI model to have BMI outliers removed and the Age model to have Age outliers removed: 
dt_without_BMIoutliers <- dt %>% anti_join(outliers_BMI)
dt_without_Ageoutliers <- dt %>% anti_join(outliers_Age)


postModel1 <- lm(charges ~ bmi, data = dt_without_BMIoutliers)  # BMI outliers removed
summary(postModel1)  # BMI predictor

postModel2 <- lm(charges ~ age, data = dt_without_Ageoutliers)  # Age outliers removed
summary(postModel2)  # Age predictor


# Recreate the diagnostics plots, with these outliers now removed:
par(mfrow = c(2,2))
plot(postModel1)     # BMI
plot(postModel2)     # Age


"
 We now have 2 cleaned data sets: 
 dt_without_BMIoutliers and dt_without_Ageoutliers
 
 These data sets will be used below to create the models
 But these sets must first be partitioned for training and test sets
"



#......................................................................................................
        ## Splitting Data 50/50 into Training and Test Sets ##

# Partition data into 50% test set:
splitPercentageBMI <- round(nrow(dt_without_BMIoutliers) %*% 0.5)
splitPercentageAge <- round(nrow(dt_without_Ageoutliers) %*% 0.5)

idxBMI <- sample(1:nrow(dt_without_BMIoutliers), splitPercentageBMI)  #sample of indices to use
idxAge <- sample(1:nrow(dt_without_Ageoutliers), splitPercentageAge)  #sample of indices to use

# Training set creation:
trainingSet_BMI <- dt_without_BMIoutliers[idxBMI, ]
trainingSet_Age <- dt_without_Ageoutliers[idxAge, ]

# Test set creation:
testSet_BMI <- dt_without_BMIoutliers[-idxBMI, ]
testSet_Age <- dt_without_Ageoutliers[-idxAge, ]



#......................................................................................................
        ## Creating the Models ##

# Model 1: Fit a simple linear regression model (charge vs bmi ) to training data:
fit1 <- lm(charges ~ bmi, trainingSet_BMI)   # BMI predictor
fit1

# Model 2: Fit a simple linear regression model (charge vs age ) to training data:
fit2 <- lm(charges ~ age, trainingSet_Age)   # Age predictor
fit2


# Training set predictions:
preds1 <- predict(fit1, trainingSet_BMI)   # for BMI
preds2 <- predict(fit2, trainingSet_Age)   # for Age


# Get RMSE for Training Data:
RMSE(y_pred= preds1, y_true= trainingSet_BMI$charges)  # model with BMI
RMSE(y_pred= preds2, y_true= trainingSet_Age$charges)  # model with Age



#......................................................................................................
        ## Apply Model to Test Data ##

# Apply to test set:
pred1_Test      <- predict(fit1, testSet_BMI)
pred2_Test      <- predict(fit2, testSet_Age)

# RMSE for the two models:
RMSE(y_pred = pred1_Test, y_true = testSet_BMI$charges)
RMSE(y_pred = pred2_Test, y_true = testSet_Age$charges)



#......................................................................................................
        ## Additional Model Performance Evaluation ##

# Summary data on models (residuals, R^2,  p-vals, etc.)
summary(fit1)  # BMI as predictor
summary(fit2)  # Age as predictor


# Visualize fit via ggplot:
# BMI model:
ggplot(dt_without_BMIoutliers, aes(bmi, charges)) +
  geom_point() +
  stat_smooth(method = lm)

# Age model:
ggplot(dt_without_Ageoutliers, aes(age, charges)) +
  geom_point() +
  stat_smooth(method = lm)



