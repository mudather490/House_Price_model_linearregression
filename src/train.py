import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# for the path url i was used from my drive
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
# We've added Postcode based on correlation analysis feedback
categorical_features = ['Suburb', 'Type', 'Method', 'Regionname']
numerical_features = ['Rooms', 'Distance', 'Propertycount', 'Postcode']

# 3. One-Hot Encoding: Convert text categories into numeric columns
X = pd.get_dummies(df_clean[categorical_features + numerical_features], columns=categorical_features)
y = df_clean['Price']

print(f"New data shape with encoded features: {X.shape}")
print(f"Included numerical features: {numerical_features}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("Model training complete.")

print(f"Intercept: {model.intercept_:,.2f}\n")

# Creating a DataFrame to sort and show the top coefficients
coef_summary = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("Top 10 Positive Factors:")
display(coef_summary.head(10))

print("\nInterpretation:")
try:
    rooms_idx = list(X.columns).index('Rooms')
    print(f"For every additional room, the price increases by approximately ${model.coef_[rooms_idx]:,.0f}.")
except ValueError:
    print("Rooms feature not found in model.")

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

# 3. Plot the regression line
plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Line')

plt.title('House Price vs Number of Rooms')
plt.xlabel('Rooms')
plt.ylabel('Price')
plt.legend()
plt.show()


# i have already trained the 'model' variable using LinearRegression in the previous cells.
# Let's confirm its performance on the test set.
y_final_pred = model.predict(X_test)
final_r2 = r2_score(y_test, y_final_pred)

print(f"Final Linear Regression R^2 Score: {final_r2:.4f}")

plt.figure(figsize=(10, 6))

# Plotting the actual vs predicted values for Linear Regression
plt.scatter(y_test, y_pred, color='green', alpha=0.3, label='Linear Regression Guesses')

# A diagonal line represents a perfect guess
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', lw=2, label='Perfect Guess Line')

plt.title('Linear Regression Performance (Actual vs. Predicted)')

plt.xlabel('Actual Price ($)')
plt.ylabel('Model Guess ($)')

plt.legend()
plt.show()

# 2. Test it using a real row from our test data
# We must use a row from X_test because it contains all 389 encoded features
sample_row = X_test.iloc[[0]]
prediction = model.predict(sample_row)

print(f"\nPrediction for house at index 0: ${prediction[0]:,.2f}")
print(f"Actual price: ${y_test.iloc[0]:,.2f}")
