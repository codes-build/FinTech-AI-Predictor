from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import numpy as np

app = Flask(__name__)
CORS(app)

# 🧠 SMART TRANSLATOR: Full names ko symbols mein convert karne ke liye
COIN_NAME_MAP = {
    "bitcoin": "btc",
    "ethereum": "eth",
    "solana": "sol",
    "dogecoin": "doge",
    "cardano": "ada",
    "ripple": "xrp",
    "shiba": "shib",
    "shiba inu": "shib",
    "polygon": "matic",
    "polkadot": "dot",
    "litecoin": "ltc"
}

@app.route('/')
def home():
    return jsonify({"message": "Omie's Universal FinTech AI Engine is LIVE!"})

@app.route('/predict/<coin_symbol>', methods=['GET'])
def predict(coin_symbol):
    try:
        # User ne jo bhi likha hai, usko clean karna
        clean_input = coin_symbol.lower().strip()
        
        # Agar full name likha hai, toh dictionary se uska short symbol nikal lo
        actual_symbol = COIN_NAME_MAP.get(clean_input, clean_input)
        
        ticker = f"{actual_symbol.upper()}-USD"
        
        # 1. Fetching market data
        df = yf.download(ticker, period="100d", interval="1d")
        
        if df.empty:
            return jsonify({"error": f"Coin '{coin_symbol.upper()}' not found in market."}), 404

        # yfinance multi-level format ko single level mein convert karna
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Features extract karna
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
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
        
        # 4. Predict Tomorrow's Price
        predicted_price = float(model.predict(latest_features)[0])
        
        # 🔥 REAL AI CONFIDENCE & SIGNAL LOGIC 🔥
        # Get predictions from all 100 individual trees in the Random Forest
        all_tree_preds = np.array([tree.predict(latest_features)[0] for tree in model.estimators_])
        # Standard deviation calculates how much the 100 trees disagree with each other
        std_dev = float(np.std(all_tree_preds))
        
        # Calculate Real Confidence % (Less disagreement = Higher Confidence)
        max_expected_dev = last_close * 0.05 # Assuming 5% dev is worst-case normal volatility
        
        if max_expected_dev > 0:
            confidence = 100 - ((std_dev / max_expected_dev) * 50)
        else:
            confidence = 50.0
            
        # Ensure confidence stays in a realistic 60% to 99% range
        real_confidence = round(max(60.0, min(99.0, confidence)), 1)

        # Calculate Real Trade Signal based on predicted percentage change
        price_diff = predicted_price - last_close
        percent_change = (price_diff / last_close) * 100

        if percent_change > 2.0:
            trade_signal = "STRONG BUY"
        elif percent_change > 0.2:
            trade_signal = "BUY"
        elif percent_change > -0.2:
            trade_signal = "HOLD"
        elif percent_change > -2.0:
            trade_signal = "SELL"
        else:
            trade_signal = "STRONG SELL"

        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        
        return jsonify({
            "coin": actual_symbol.upper(),
            "current_date": today.strftime("%Y-%m-%d"),
            "target_date": tomorrow.strftime("%Y-%m-%d"),
            "last_close_price": round(last_close, 4),
            "predicted_price": round(predicted_price, 4),
            "confidence_score": real_confidence,
            "trade_signal": trade_signal
        })

    except Exception as e:
        print(f"Error Backend: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
