# backtest/debug_data.py
import pandas as pd
from app.io.storage import load_cont
from app.indicators.features import compute_features

symbol = "BANKNIFTY"   # change to NIFTY to test
tf = "5m"              # 1m / 3m / 5m

df = load_cont(symbol, tf=tf)
print("Loaded bars:", df.shape, "range:", df.index.min(), "→", df.index.max())

feats = compute_features(df, add_vwap=True).dropna(subset=["rsi","atr","ema20","ema50","pp","tc","bc"], how="any")
print("Features cols:", feats.columns.tolist())
print("NaN counts (key cols):")
print(feats[["rsi","atr","ema20","ema50","pp","tc","bc","vwap"]].isna().sum())

# How many rows survive typical warmup?
warm = feats.dropna(subset=["rsi","ema20","ema50"])
print("Rows after warmup (rsi/ema):", warm.shape)

# See distributions to gauge thresholds
print("RSI quantiles:\n", warm["rsi"].quantile([0.1,0.25,0.5,0.75,0.9]))
print("EMA trend %:", (warm["ema20"] > warm["ema50"]).mean()*100, "%")

print("\nSample last 5 rows:")
print(warm[["c","rsi","ema20","ema50","vwap","tc"]].tail())
