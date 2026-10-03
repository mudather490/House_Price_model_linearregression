# 🏠 House Price Prediction with Linear Regression

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Baseline%20Model-green)

An end-to-end Machine Learning project that predicts house prices from property features using the **Melbourne Housing dataset** (13,000+ rows). The goal was not only to train a model, but to practice the full workflow an ML engineer follows: loading data, EDA, feature selection, evaluation, visualization, and saving the model.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| R² Score | **0.6462** |
| Mean Squared Error (MSE) | 121,211,553,932.57 |
| Root Mean Squared Error (RMSE) | ≈ 348,000 |

The model explains about **64.6%** of the variation in house prices. This is a solid baseline. The RMSE (the square root of the MSE) shows that predictions are typically off by roughly 348K, which leaves clear room for improvement with better features and stronger models.

<!-- Add your plots here. Make sure the file names match your images/ folder. -->
### Actual vs. Predicted Prices

![Actual vs Predicted](images/actual_vs_predicted.png)

### Regression Plot

![Regression Plot](images/regression_plot.png)

---

## 🎯 Problem Statement

House prices depend on many factors, such as the number of rooms, distance from the city center, and the size of the property. The objective is to train a model that estimates the price of a house from these features.

- **Problem type:** Regression
- **Target column:** `Price`

---

## 📁 Dataset

- **Source:** Melbourne Housing dataset (`MELBOURNE_HOUSE_PRICES_LESS.csv`)
- **Rows:** ~13,000+
- **Features used:** numerical property features

| Feature | Description |
|---------|-------------|
| `Rooms` | Number of rooms |
| `Distance` | Distance from the city center |
| `Propertycount` | Number of properties in the suburb |

<!-- Add every other feature you used in training, so this table matches your example prediction. -->

---

## 🔄 Workflow

1. Load the dataset
2. Exploratory Data Analysis (EDA)
3. Check missing values and duplicates
4. Analyze feature correlations
5. Select numerical features
6. Split into training and testing sets
7. Train a Linear Regression model
8. Evaluate with MSE and R²
9. Visualize predictions
10. Save the trained model with `joblib`

---

## 🛠 Tech Stack

- **Language:** Python
- **Libraries:** NumPy, Pandas, Matplotlib, scikit-learn, joblib
- **Environment:** Jupyter Notebook

---

## 📂 Project Structure

```
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

## 🚀 Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/mudather490/house-price-prediction.git
cd house-price-prediction
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Train the model**

```bash
python src/train.py
```

**4. Make predictions**

```bash
python src/predict.py
```

### Example: predicting a new house

```python
import joblib

model = joblib.load("models/house_price_model.pkl")

# Values must be in the same order as the features used in training
new_house = [[4, 2, 6.5, 2, 180, 450]]
prediction = model.predict(new_house)

print(f"Predicted price: {prediction[0]:,.0f}")
```

---

## 🧠 Development Process

I followed a **learning-first approach** instead of relying on AI to write the project for me.

**Version 1: built by me.** I wrote the first version on my own: loading and cleaning the data, checking duplicates, exploring correlations, selecting features, training the model, interpreting coefficients, and predicting a new house. I made mistakes along the way, such as training on only one feature, which gave weak performance.

**Review and improvement.** After finishing, I used AI as a **mentor and code reviewer, not a code generator**. It helped me review my code, understand my mistakes, improve the project structure, and see why some decisions are better than others.

**Version 2: rebuilt.** I then rebuilt the project with what I had learned: more features, a cleaner structure, and proper evaluation.

---

## 📚 What I Learned

- Data loading, inspection, and cleaning
- Feature selection and correlation analysis
- Train/test split
- Linear Regression and interpreting its coefficients
- Evaluating models with MSE and R²
- Saving and reusing a trained model
- Organizing an ML project like production code

---

## ⚠️ Limitations

- Linear Regression assumes a linear relationship between features and price, which is rarely fully true for housing.
- Outliers (very expensive properties) can pull the model's predictions off.
- Only numerical features are used, so information such as suburb and property type is ignored.

---

## 🔮 Future Improvements

- [ ] Feature engineering (including categorical features)
- [ ] More careful outlier handling
- [ ] Compare against Decision Tree and Random Forest models
- [ ] Hyperparameter tuning
- [ ] Build a prediction API with FastAPI
- [ ] Deploy the model to the cloud

---

## 👤 Author

**Mudather Kbyer**
Learning Machine Learning by building real projects.

- GitHub: [@mudather490](https://github.com/mudather490)
- LinkedIn: [linkedin.com/in/mudaxkbyer](https://www.linkedin.com/in/mudaxkbyer)
