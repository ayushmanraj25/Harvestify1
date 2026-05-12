import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load data
df = pd.read_csv("Data-raw/irrigation.csv")

# Encode
df = pd.get_dummies(df, columns=['Crop', 'Soil'])

# Features & target
X = df.drop('Water', axis=1)
y = df['Water']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
pickle.dump(model, open("models/irrigation_model.pkl", "wb"))

# 🔥 IMPORTANT: save columns
pickle.dump(X.columns.tolist(), open("models/irrigation_columns.pkl", "wb"))

print("Model trained + columns saved ✅")