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
* `Bathroom`
* `Distance`
* `Car`
* `BuildingArea`
* `Landsize`

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

Evaluation metrics:

* **Mean Squared Error (MSE):** Computed on the test set.
* **R² Score:** Measures how much variance in house prices is explained by the model.

The model achieved a reasonable baseline performance and demonstrated the full regression workflow.

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
