import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# 1. Define mixed assets (Crypto + Stocks)
tickers = ["BTC-USD", "SOL-USD", "AAPL", "TSLA"]
start_date = "2021-01-01"  # Solana ka data 2020 mid ke baad se zyada stable hai
end_date = "2026-09-03"

print(f"Fetching mixed market data for {tickers}...")
data = yf.download(tickers, start=start_date, end=end_date)

# 2. Isolate 'Close' prices
df_close = data['Close']

# 3. Handle the 'Weekend Gap' Problem
# ffill (forward-fill) Friday ke stock prices ko Sat/Sun par copy kar dega
# bfill (backward-fill) ensure karta hai ki agar shuruat mein koi data miss ho toh fill ho jaye
df_close = df_close.ffill().bfill()

# 4. Normalize the data for the AI Model
# Normalize values between 0 and 1 so BTC ($60k+) and SOL ($140+) scale equally with AAPL/TSLA
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_close)

# Rebuild the DataFrame with the scaled data
df_scaled = pd.DataFrame(scaled_data, columns=df_close.columns, index=df_close.index)

# 5. Save locally for Step 3 (Model Training)
os.makedirs("data", exist_ok=True)
df_close.to_csv("data/mixed_raw_prices.csv")
df_scaled.to_csv("data/mixed_scaled_prices.csv")

print("Hybrid data successfully fetched, gap-filled, scaled, and saved!") 