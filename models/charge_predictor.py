import json

import pandas as pd
import joblib

# Load trained model
model = joblib.load('insurance_model_new.pkl')  # Adjust if needed

# with open('../../models/model_columns.json', 'r') as f:
#     expected_columns = json.load(f)
# New sample input
new_data = pd.DataFrame({

    'age': [28],
    'sex': [0],
    'bmi': [26.2],
    'children': [2],
    'smoker': [1],
    'region': ['northwest'],
    'risk_score': [4.5]
})

# One-hot encode
# new_data = pd.get_dummies(new_data, columns=['sex', 'smoker', 'region'])

# These must exactly match the features your model was trained with
expected_columns = [
    'age', 'sex', 'bmi', 'children',
    'smoker',
    'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest',
    'risk_score',
]

# Add any missing columns (set to 0)
for col in expected_columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Reorder columns
new_data = new_data[expected_columns]

# Predict
prediction = model.predict(new_data)
print(f"Predicted charges: {prediction[0]:.2f}")
