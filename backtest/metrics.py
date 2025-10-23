# backtest/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

def summarize_trades(trades: pd.DataFrame) -> dict:
    closes = trades[trades["pnl"] != 0.0]
    n_trades = len(closes)
    wins = (closes["pnl"] > 0).sum()
    losses = (closes["pnl"] <= 0).sum()
    total_pnl = closes["pnl"].sum()
    avg_win = closes.loc[closes["pnl"] > 0, "pnl"].mean() if wins else 0.0
    avg_loss = closes.loc[closes["pnl"] <= 0, "pnl"].mean() if losses else 0.0

    # daily metrics
    if not closes.empty:
        daily = closes["pnl"].groupby(closes.index.date).sum()
        sharpe = (daily.mean() / (daily.std() + 1e-9)) * np.sqrt(252)
        max_dd = _max_drawdown(closes["pnl"].cumsum())
    else:
        sharpe, max_dd = 0.0, 0.0

    return {
        "trades": int(n_trades),
        "win_rate": float(wins / n_trades * 100) if n_trades else 0.0,
        "total_pnl": float(total_pnl),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "sharpe_daily252": float(sharpe),
        "max_drawdown": float(max_dd),
    }

def _max_drawdown(equity_series: pd.Series) -> float:
    cum = equity_series.cumsum() if equity_series.index.equals(range(len(equity_series))) else equity_series
    peak = cum.cummax()
    dd = cum - peak
    return float(dd.min())
