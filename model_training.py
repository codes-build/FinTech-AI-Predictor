import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 1. Load the scaled data from Step 2
df = pd.read_csv("data/mixed_scaled_prices.csv", index_col=0)
target_data = df[['SOL-USD']].values

# 2. Create Sequences (Use past 60 days to predict the 61st day)
X, y = [], []
time_steps = 60

for i in range(time_steps, len(target_data)):
    # Flatten the 60 days data into a single row for Random Forest
    X.append(target_data[i-time_steps:i, 0])
    y.append(target_data[i, 0])

X, y = np.array(X), np.array(y)

# 3. Split into Training (80%) and Testing (20%) data
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]

print(f"Building AI Brain... Training on {X_train.shape[0]} days of data.")

# 4. Build and Train the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)

print("Training started... this will be super fast on your Mac M5!")
model.fit(X_train, y_train)

# 5. Save the trained model for our Website
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/solana_rf_model.pkl")

print("Success! AI Model is trained and saved as 'solana_rf_model.pkl' in the models folder.")