import joblib
import pandas as pd
import numpy as np

# 1. Load the model
model_path = 'house_price_model.pkl'
trained_model = joblib.load(model_path)

# 2. Prepare your input data
# Since the model expects 389 features, we start with a template of zeros
# based on the training columns (X.columns)
new_house = pd.DataFrame(0, index=[0], columns=X.columns)

# Fill in the known details
new_house['Rooms'] = 3
new_house['Distance'] = 12.5
new_house['Propertycount'] = 4500

# Set the categorical features (e.g., Suburb_Abbotsford = 1)
# Note: You must pick from the columns created during get_dummies
if 'Suburb_Abbotsford' in new_house.columns:
    new_house['Suburb_Abbotsford'] = 1
if 'Type_h' in new_house.columns:
    new_house['Type_h'] = 1

# 3. Predict!
prediction = trained_model.predict(new_house)

print(f"The predicted price for this house is: ${prediction[0]:,.2f}")
