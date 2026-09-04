from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Omie's Universal FinTech AI Engine is LIVE!"})

@app.route('/predict/<coin_symbol>', methods=['GET'])
def predict(coin_symbol):
    try:
        # User jo bhi symbol de (e.g., doge, sol), uske aage '-USD' lagakar live data uthao
        ticker = f"{coin_symbol.upper()}-USD"
        
        # 150 days ka data fetch karna
        df = yf.download(ticker, period="150d", interval="1d")
        
        if df.empty:
            return jsonify({"error": f"Coin symbol '{coin_symbol.upper()}' not found. Try BTC, SOL, DOGE, etc."}), 404

        # Advanced Pro-Trader Logic
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_14'] = df['Close'].rolling(window=14).mean()
        df['Volatility'] = df['Close'].rolling(window=7).std()
        df['Target'] = df['Close'].shift(-1)
        
        df.dropna(inplace=True)
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_7', 'SMA_14', 'Volatility']
        X = df[features].values
        y = df['Target'].values
        
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, y)
        
        last_row = df.iloc[-1]
        latest_features = np.array([[
            last_row['Open'], last_row['High'], last_row['Low'], 
            last_row['Close'], last_row['Volume'], 
            last_row['SMA_7'], last_row['SMA_14'], last_row['Volatility']
        ]])
        predicted_price = model.predict(latest_features)[0]
        
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        
        return jsonify({
            "coin": coin_symbol.upper(),
            "current_date": today.strftime("%Y-%m-%d"),
            "target_date": tomorrow.strftime("%Y-%m-%d"),
            "last_close_price": round(float(last_row['Close']), 4), # Chote coins (jaise DOGE) ke liye 4 decimals
            "predicted_price": round(float(predicted_price), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
