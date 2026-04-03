# Titanic Survival Prediction Mini-Project

## Goal
Build a documented Python pipeline that loads the classic Titanic dataset, performs data cleaning and feature engineering, trains machine learning models, and exposes a function to predict a passenger's probability of survival based on their attributes.

## Project Overview
This project takes raw passenger data from the infamous Titanic shipwreck and uses it to train machine learning models to predict who survived and who didn't. It compares two classification algorithms: **Logistic Regression** and **K-Nearest Neighbors (KNN)**.

### Key Features
* **Exploratory Data Analysis (EDA):** Generates descriptive statistics, missing value checks, and histograms.
* **Data Cleaning:** Handles missing values (e.g., imputing median age, filling 'Embarked' ports) and drops heavily incomplete columns like `Cabin`.
* **Feature Engineering:** * Converts `Sex` to a binary numeric format.
  * Groups `Fare` into a binary category (above/below median).
  * Creates a new `FamilySize` feature by combining siblings/spouses (`SibSp`) and parents/children (`Parch`).
  * Extracts and encodes `TicketPrefix`.
* **Modeling & Evaluation:** * Scales features using `StandardScaler`.
  * Trains a Logistic Regression model and evaluates it using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and an ROC Curve.
  * Trains a KNN model and loops through different $k$ values (1-25) to find the optimal number of neighbors (peaking around $k=12$).
* **Prediction Function:** Includes a custom `predict_survival` function that accepts new, unseen passenger data and outputs the survival prediction and probability.

## Setup and Installation

1. **Clone or download this repository.**
2. **Ensure you have the Titanic dataset.** Download `Titanic-Dataset.csv` and ensure the file path in the script matches its location on your machine.
3. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt
