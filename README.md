# Tractor-Pricing


# Overview

This project was aimed at understanding the importance of data cleaning and feature engineering when collecting datasets.  We were tasked with understanding which features were the most important predictors in determining the price of tractors and other heavy machinery at auction.  The data had a lot of null values as well as being in various formats because it was taken from various sources.  In our group we learned to triage tasks and spend a good majority of the time allotted before we had to present our findings to feature engineering based on domain knowledge or intuition about what might effect pricing. We then developed a baseline Linear regression model to predict the average selling price of the tractors.  Then as added new features we re-evaluated our model and kept the features that performed well.  The final model predicts the selling price of a tractor based on the year made and the product size.  The data for this case study was not made public.


# Motivation

The motivation behind this problem was to use various regression models to predict the sales price of tractors and experience dealing with the reality that is messy data.  It was important to practice data cleaning and gain intuition for feature engineering because those are two common practices of a Data Scientist.  This was also one of our first case studies where we had a limited amount of time to gain insight into a problem, create a model, and present our results.


# Data

The data set was taken from 3 different auction houses with about 54 features.  These features included both categorical and numerical data while also having many columns that were in varying format as well as numerous null entries.  We cross validated our training set while also being provided with a test set.


# Data Cleaning and Feature Engineering

Data had many null values, so one of the biggest issues was getting features to put into our regression model.  Some of the features we decided to use were year made and product size because we thought intuitively that these features might provide the most signal as well as that data was fairly clean.  We also tried creating dummy variables for a few of the features we thought might increase the predictive power of our model and tried to get signal from various null values, but that did not work well so we did not include those features in our final models.  


# Model

We decided to use a Linear Regression model. For a baseline model we basically predicted the mean of all the tractor prices and benchmarked from there.  We decided to use RMSE as our metric for benchmarking and iterated through multiple combinations of features to identify our best model.  The models that we ended up keeping had the features year made with the type of hydraulics and year made with product size.


# Result and Inference

From a data science perspective, we understood certain limitations of the regression model as well as its ease of interpretability.  We also got a picture of how messy real world data could be, and how to approach cleaning said data, and how important featuring engineering can be in producing models with good predictive power.
