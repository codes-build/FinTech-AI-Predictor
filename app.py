from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app) # Yeh hamari aane wali React/HTML website ko block nahi hone dega

# 1. Load the trained AI model
model = joblib.load("models/solana_rf_model.pkl")

@app.route('/')
def home():
    return jsonify({"message": "Omie's FinTech AI Backend is LIVE!"})

@app.route('/predict/solana')
def predict_solana():
    try:
        raw_df = pd.read_csv("data/mixed_raw_prices.csv", index_col=0)
        solana_raw = raw_df[['SOL-USD']]
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        solana_scaled = scaler.fit_transform(solana_raw)
        
        last_60_days = solana_scaled[-60:]
        X_input = np.array([last_60_days.flatten()])
        
        predicted_scaled = model.predict(X_input)
        predicted_actual = scaler.inverse_transform([[predicted_scaled[0]]])
        last_actual_price = solana_raw.iloc[-1, 0]

        # NEW: Get last 30 days data for the chart
        last_30_dates = solana_raw.index[-30:].tolist()
        last_30_prices = solana_raw['SOL-USD'][-30:].tolist()
        
        return jsonify({
            "asset": "Solana (SOL-USD)",
            "last_close_price": round(float(last_actual_price), 2),
            "predicted_next_day_price": round(float(predicted_actual[0][0]), 2),
            "history_dates": last_30_dates,
            "history_prices": last_30_prices
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Port 5000 par server run hoga
    app.run(debug=True, port=5000)