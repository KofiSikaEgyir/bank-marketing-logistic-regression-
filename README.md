# Logistic Regression Model for Bank Marketing Prediction

## Project Overview

This project applies a logistic regression model to a banking marketing dataset to predict whether a client subscribed to a term deposit. The target variable is `y`, where `1` represents clients who subscribed and `0` represents clients who did not subscribe.

The aim of the project is to use machine learning techniques to understand customer subscription behaviour and build a classification model that can predict the likelihood of a customer responding positively to a bank marketing campaign.

## Dataset

The dataset used is the Bank Marketing dataset. It contains customer demographic information, banking-related attributes, campaign contact details, and economic indicators.

Some of the key variables include:

- `age`: Age of the client
- `job`: Type of job
- `marital`: Marital status
- `education`: Level of education
- `default`: Whether the client has credit in default
- `housing`: Whether the client has a housing loan
- `loan`: Whether the client has a personal loan
- `contact`: Type of contact communication
- `month`: Last contact month
- `day_of_week`: Last contact day of the week
- `previous`: Number of previous contacts before the current campaign
- `poutcome`: Outcome of the previous marketing campaign
- `euribor3m`: Euribor 3-month rate
- `y`: Target variable showing whether the client subscribed to a term deposit

## What Was Done

The project started by importing the necessary Python libraries, including `pandas`, `numpy`, `matplotlib`, `seaborn`, 
and machine learning tools from `scikit-learn`.

The dataset was loaded and inspected to understand its structure, number of observations, and available variables. 
Missing values were removed to ensure the model was trained on complete records.

The `education` variable was cleaned by combining `basic.4y`, `basic.6y`, and `basic.9y` into one category called `Basic`. 
This reduced the number of categories and made the variable easier to use in modelling.

Exploratory data analysis was then carried out using count plots and cross-tabulation charts. 
These visualizations helped show how customer subscription differed across job type, marital status, education level, month of contact, and day of the week.

Since machine learning models require numerical input, categorical variables 
such as `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, and `poutcome` 
were converted into dummy variables.

Recursive Feature Elimination (RFE) was used to select the most useful predictor variables for the logistic regression model.
After feature selection, the final variables used in the model included:

- `previous`
- `euribor3m`
- `job_blue-collar`
- `job_retired`
- `job_services`
- `job_student`
- `default_no`
- `month_aug`
- `month_dec`
- `month_jul`
- `month_nov`
- `month_oct`
- `month_sep`
- `day_of_week_fri`
- `day_of_week_wed`
- `poutcome_failure`
- `poutcome_nonexistent`
- `poutcome_success`

The data was split into training and testing sets using a 70 percent training and 30 percent testing split. 
A logistic regression model was then trained on the training data and evaluated on the test data.

## Model Used

The main model used in this project is:

```python
LogisticRegression()
