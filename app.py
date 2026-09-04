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
        
        # 1. Fetching market data
        df = yf.download(ticker, period="100d", interval="1d")
        
        if df.empty:
            return jsonify({"error": f"Coin symbol '{coin_symbol.upper()}' not found. Try BTC, SOL, DOGE, etc."}), 404

        # FIX 1: yfinance multi-level format ko single level mein convert karna
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # FIX 2: Aaj ka data safe jagah store karna (BEFORE dropping NA)
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # .iloc[-1:] hamesha ek 2D array return karega, 3D error kabhi nahi aayegi
        latest_features = df[features].iloc[-1:].values 
        last_close = float(df['Close'].iloc[-1])

        # 2. Preparing data for AI Training
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
        
        X = df[features].values
        y = df['Target'].values
        
        # 3. Stable Random Forest AI Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 4. Predict Tomorrow's Price accurately
        predicted_price = model.predict(latest_features)[0]
        
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        
        return jsonify({
            "coin": coin_symbol.upper(),
            "current_date": today.strftime("%Y-%m-%d"),
            "target_date": tomorrow.strftime("%Y-%m-%d"),
            "last_close_price": round(last_close, 4),
            "predicted_price": round(float(predicted_price), 4)
        })

    except Exception as e:
        print(f"Error Backend: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
