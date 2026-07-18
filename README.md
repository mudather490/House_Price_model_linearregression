# House Price Prediction using Linear Regression

## 1. Project Overview

This is one of my first end-to-end Machine Learning projects. In this project, I built a Linear Regression model to predict house prices using housing features from the Melbourne housing dataset.

The goal of the project was not only to train a model, but also to practice the complete Machine Learning workflow used by ML engineers, including data loading, EDA, feature selection, model evaluation, visualization, and model saving.

---

## 2. Dataset

* **Dataset:** Melbourne Housing Dataset
* **Target Column:** `Price`
* **Problem Type:** Regression
* **Rows:** ~13k+
* **Features Used:** Numerical housing features related to property characteristics.

---

## 3. Problem Statement

House prices are influenced by many factors such as the number of rooms, bathrooms, distance from the city, and building area. The objective of this project is to train a model that can estimate the price of a house based on these features.

---

## 4. Features

Features used in the model include:

* `Rooms`
* `Properityacont`
* `Distance`

Target:

* `Price`

---

## 5. Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import joblib
```

---

## 6. Project Workflow

1. Load the dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Check missing values and duplicates.
4. Analyze feature correlations.
5. Select numerical features.
6. Split the dataset into training and testing sets.
7. Train a Linear Regression model.
8. Evaluate the model using MSE and R².
9. Visualize predictions.
10. Save the trained model using `joblib`.

---

## 7. Model Performance


## Model Performance

- Mean Squared Error (MSE): 121,211,553,932.57
- R² Score: 0.6462

The Linear Regression model explained approximately 64.6% of the variation in house prices. This provides a solid baseline model, and future work will focus on feature engineering and experimenting with more advanced regression algorithms to improve predictive performance.

---

## 8. Results

The project successfully:

* Trained a Linear Regression model.
* Generated house price predictions.
* Evaluated model quality using MSE and R².
* Visualized the relationship between actual and predicted prices.
* Saved the trained model for future use.

Example prediction:

```python
new_house = [[4, 2, 6.5, 2, 180, 450]]
prediction = model.predict(new_house)
```

---

## 9. Folder Structure

```text
house-price-prediction/
│
├── data/
│   └── MELBOURNE_HOUSE_PRICES_LESS.csv
│
├── notebooks/
│   └── house_price_analysis.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   └── house_price_model.pkl
│
├── images/
│   ├── regression_plot.png
│   └── actual_vs_predicted.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 10. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python src/train.py
```

### Make predictions

```bash
python src/predict.py
```

---

## 11. Future Improvements

Future improvements I plan to add:

* Feature engineering.
* Handling outliers more carefully.
* Comparing Linear Regression with Decision Tree and Random Forest models.
* Hyperparameter tuning.
* Building a simple prediction API with FastAPI.
* Deploying the model to the cloud.

---

## About Me

I am building my Machine Learning portfolio by implementing projects from scratch, focusing on understanding the workflow, evaluating models properly, and writing production-style code rather than only completing courses.

# Development Process

This project was built as part of my Machine Learning learning journey.

Instead of relying on AI to build the project for me, I followed a learning-first approach.

### Version 1 (Built by Me)

I wrote the first version of the project completely on my own. My workflow was:

1. Import libraries
2. Load the dataset
3. Check for missing values
4. Clean the data
5. Check for duplicate values
6. Explore feature correlations
7. Select features
8. Split the dataset into training and testing sets
9. Train a Linear Regression model
10. Make predictions
11. Interpret the model coefficients
12. Predict the price of a new house
13. Evaluate the model

During this first version, I made several mistakes, such as using only one feature for training, which resulted in lower model performance.

### Learning and Improvement

After completing the project, I used AI as a mentor and code reviewer—not as a code generator.

AI helped me:

* Review my code
* Explain my mistakes
* Improve the project structure
* Apply better Machine Learning practices
* Understand why some decisions were better than others

I then rebuilt and improved the project based on what I had learned.

### What I Learned

Through this project, I gained practical experience with:

* Data loading and inspection
* Data cleaning
* Feature selection
* Train/Test Split
* Linear Regression
* Model evaluation using MSE and R² Score
* Model interpretation
* Making predictions on new data

This project reflects my own learning process and demonstrates how I use AI to improve my understanding rather than to replace it.

