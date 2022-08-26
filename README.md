# customerchurn
*Phase 3 Solo Project - Telecom Customer Churn

**Objective: Create a predictive model for customer churn True or False for SyriaTel telecom

**Business Problem: 
* Maximize Accuracy and Recall to identify all churn customers with regularly collected telecom phone plan data. Use these predictions to target customer retention methods with the goal of stopping cancellations

**Data:

****Source: https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset

****Predictors:
* account length
* international plan
* voice mail plan
* number of voice mail messages
* total day minutes used
* day calls made
* total day charge
* total evening minutes
* total evening calls
* total evening charge
* total night minutes
* total night calls
* total night charge
* total international minutes used
* total international calls made
* total international charge
* number customer service calls made

****Target:
*churn 

**EDA:
**** Columns to Drop:
* area code - only 3 unique values, not meaningful for predictive context
* phone number - only serves as a uniqueid and is redundant

**Base Models (No Hyperparameter Tuning):
****ModelWithCV class: 
* Class for saving models and running cross validation summaries

****Baseline:
* Dummy model: ~85% accuracy (14.4% Churn)

****Logistic Regression:
* 86% accuracy - not much of an improvement from baseline

****DecisionTree
* 90% accuracy 

****KNearestNeighbors
* 89% accuracy

****RandomForest
* 94% accuracy

****GradientBoosting
* 95% accuracy

** Models w/ GridSearch
* Imbalanced Data: Low number of 'churn' might be affecting model performance. Used SMOTE to overrepresent minority data ('churn') 
* GridSearch: Using dictionaries (key=hyperparameter, value=options for parameters) to run through iterations of models to find best performing ones
* Eventually with Hyperparameter tuning, GradientBoosting performed the best with 95% accuracy, 78% recall

![Alt](/images/confusion_heatmap.png "Confusion Matrix GB")