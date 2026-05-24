🚀 Predicting Kickstarter Campaign Success

Using Pre-Launch and Early Campaign Data

📌 Project Overview

This project builds machine learning models to predict whether a Kickstarter campaign will be successful or failed, using information available before launch and early in the campaign lifecycle.

The goal is to simulate real-world decision-making for creators, investors, and platforms, where final outcomes are unknown and only partial data is available.

🎯 Business Problem

Kickstarter creators face two key decisions:

Before launch:
Is this campaign likely to succeed given its goal, category, and timing?

Early in the campaign:
Based on early performance, is this campaign on track to succeed?

This project addresses both questions by comparing:

Pre-launch prediction models

Early-campaign prediction models

📊 Dataset

Source: Kaggle – Kickstarter Projects
The dataset contains historical Kickstarter campaigns with features including:

Campaign goal

Amount pledged

Number of backers

Campaign category and subcategory

Country and currency

Launch and deadline dates

Final campaign outcome (successful, failed, etc.)

Only campaigns with final outcomes of successful or failed are included in this analysis.

🧠 Target Variable

Campaign Success

1 → Successful (goal met or exceeded)

0 → Failed

This is treated as a binary classification problem.

⏱️ Modeling Phases
Phase 1: Pre-Launch Prediction

Uses only information available before the campaign launches.

Planned Features:

Campaign goal (log-transformed)

Main category / subcategory

Campaign duration

Launch month and weekday

Country and currency

Objective:
Estimate the probability of campaign success before launch.

Phase 2: Early-Campaign Prediction

Adds early performance signals observed shortly after launch.

Planned Features:

Early pledged amount

Early backer count

Percentage of goal reached early

Funding velocity (pledged per day)

Objective:
Measure how much predictive power early campaign data adds compared to pre-launch features alone.

🛠️ Methodology

Exploratory Data Analysis (EDA)

Data cleaning and preprocessing

Feature engineering

Baseline models for comparison

Logistic Regression for interpretability

Tree-based models for improved performance

Cross-validation for model robustness

Evaluation using:

ROC-AUC

Precision and Recall

Confusion Matrix

📈 Results (In Progress)

This section will be updated after model training and evaluation.

Planned analysis includes:

Comparison of pre-launch vs early-campaign model performance

Identification of the most influential features

Discussion of false positives vs false negatives

Tradeoffs between model interpretability and accuracy

📊 Dashboard (Planned)

An interactive Streamlit dashboard will allow users to:

Input campaign details

Receive a predicted probability of success

Compare pre-launch and early-campaign predictions

Explore feature importance and model behavior

🚨 Limitations

Early-campaign features are approximated due to dataset constraints

External factors such as marketing, social media, and creator reputation are not included

Predictions are based on historical Kickstarter data and may not generalize to future trends

🧠 Key Takeaways (To Be Completed)

Expected insights include:

The degree to which campaign success can be predicted before launch

The impact of early funding signals on prediction accuracy

The importance of feature engineering and validation in realistic ML projects

📂 Project Structure
kickstarter-success-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── clean_data.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── evaluation.py
│
├── reports/
│   ├── figures/
│   └── dashboard/
│
├── notebooks/
│   └── eda.ipynb
│
├── app.py
└── README.md

✅ Project Status

✅ Dataset selected

✅ Project scope defined

 Data inspection and cleaning

 Exploratory data analysis

 Pre-launch model

 Early-campaign model

 Dashboard

 Final report and polish