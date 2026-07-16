import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Placeholder for data loading
path ='/content/drive/MyDrive/Datasate/MELBOURNE_HOUSE_PRICES_LESS.csv'
df = pd.read_csv(path)
display(df.head())

print("Please load your dataset in this cell.")

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Basic statistics
display(df.describe())


# 1. Drop rows where Price is missing
df_clean = df.dropna(subset=['Price'])

# 2. Select more features, including categorical ones
# We'll use Suburb, Type, and Method as well as our numbers
categorical_features = ['Suburb', 'Type', 'Method', 'Regionname']
numerical_features = ['Rooms', 'Distance', 'Propertycount']

# 3. One-Hot Encoding: Convert text categories into numeric columns
X = pd.get_dummies(df_clean[categorical_features + numerical_features], columns=categorical_features)
y = df_clean['Price']

print(f"New data shape with encoded features: {X.shape}")
print("We went from 3 features to hundreds of specific 'clues'!")









X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("Model training complete.")

print(f"Intercept: {model.intercept_:,.2f}\n")

# We use X.columns because 'features' was not defined, 
# and X.columns contains all 389 encoded feature names.
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:,.2f}")

print("\nInterpretation:")
# model.coef_[0] corresponds to the first column in X, which is 'Rooms'
print(f"For every additional room, the price increases by approximately ${model.coef_[0]:,.0f}.")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.4f}")

# Visualization - Improved sorting
plt.figure(figsize=(10, 6))

# 1. Scatter plot of actual data
plt.scatter(X_test['Rooms'], y_test, color='blue', alpha=0.3, label='Actual Prices')

# 2. Sort the test data for a smooth regression line
sorted_index = X_test['Rooms'].argsort()
X_sorted = X_test['Rooms'].iloc[sorted_index]
y_pred_sorted = y_pred[sorted_index]


# We have already trained the 'model' variable using LinearRegression in the previous cells.
# Let's confirm its performance on the test set.
y_final_pred = model.predict(X_test)
final_r2 = r2_score(y_test, y_final_pred)

print(f"Final Linear Regression R^2 Score: {final_r2:.4f}")

plt.figure(figsize=(10, 6))

# Plotting the actual vs predicted values for Linear Regression
plt.scatter(y_test, y_pred, color='green', alpha=0.3, label='Linear Regression Guesses')


#1 . Save the Linear Regression model
joblib.dump(model, "house_price_model.pkl")
print("Linear Regression model saved to house_price_model.pkl")

# 2. Test it using a real row from our test data
# We must use a row from X_test because it contains all 389 encoded features
sample_row = X_test.iloc[[0]]
prediction = model.predict(sample_row)

print(f"\nPrediction for house at index 0: ${prediction[0]:,.2f}")
print(f"Actual price: ${y_test.iloc[0]:,.2f}")









