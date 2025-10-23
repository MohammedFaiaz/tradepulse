# backtest/run_backtest.py
from __future__ import annotations
import argparse
import pandas as pd
from app.io.storage import load_cont
from app.indicators.features import compute_features
from backtest.engine import Backtester, BTConfig
from backtest.metrics import summarize_trades

def parse_args():
    ap = argparse.ArgumentParser(description="Run simple intraday backtest on continuous futures.")
    ap.add_argument("--symbol", default="BANKNIFTY", help="BANKNIFTY or NIFTY")
    ap.add_argument("--tf", default="5m", help="1m, 3m, or 5m")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--entry_rsi", type=float, default=60.0)
    ap.add_argument("--exit_rsi", type=float, default=50.0)
    ap.add_argument("--cost", type=float, default=50.0)
    ap.add_argument("--slip", type=float, default=0.0)
    return ap.parse_args()

def main():
    args = parse_args()

    df = load_cont(args.symbol, tf=args.tf)
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end)]

    # Compute features (relaxed: only require RSI/EMAs warm-up)
    feats = compute_features(df, add_vwap=True)
    feats = feats.dropna(subset=["rsi","ema20","ema50"], how="any")

    cfg = BTConfig(
        entry_rsi=args.entry_rsi,
        exit_rsi=args.exit_rsi,
        cost_per_trade=args.cost,
        slippage_points=args.slip,
    )
    bt = Backtester(cfg)
    trades = bt.run(feats)

    summary = summarize_trades(trades)
    print("\n=== Backtest Summary ===")
    for k, v in summary.items():
        print(f"{k:>16}: {v}")

    print("\nLast 10 trade rows:")
    print(trades.tail(10))

if __name__ == "__main__":
    main()
