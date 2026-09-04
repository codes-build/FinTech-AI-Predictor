from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Omie's Universal FinTech AI Engine is LIVE!"})

@app.route('/predict/<coin_symbol>', methods=['GET'])
def predict(coin_symbol):
    try:
        ticker = f"{coin_symbol.upper()}-USD"
        
        # Fetching 100 days of market data
        df = yf.download(ticker, period="100d", interval="1d")
        
        if df.empty:
            return jsonify({"error": f"Coin symbol '{coin_symbol.upper()}' not found. Try BTC, SOL, DOGE, etc."}), 404

        # Preparing data
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
        
        X = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = df['Target'].values
        
        # Stable Random Forest AI Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Predict Tomorrow's Price using TODAY'S closing data
        last_row = df.iloc[-1]
        latest_features = np.array([[
            last_row['Open'], last_row['High'], last_row['Low'], 
            last_row['Close'], last_row['Volume']
        ]])
        predicted_price = model.predict(latest_features)[0]
        
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        
        return jsonify({
            "coin": coin_symbol.upper(),
            "current_date": today.strftime("%Y-%m-%d"),
            "target_date": tomorrow.strftime("%Y-%m-%d"),
            "last_close_price": round(float(last_row['Close']), 4),
            "predicted_price": round(float(predicted_price), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
