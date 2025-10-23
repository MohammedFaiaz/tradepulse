# backtest_strategies/walkforward_rsi.py
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# ---- Project paths (match your setup) ----
DATA_CONT = Path(r"C:\tradePulse\data\continuous")

# ---- Features (lean copy of your quick_backtest logic) ----
def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Guard optional cols
    for col in ["v","oi","series","data_source"]:
        if col not in out.columns: out[col] = np.nan
    # RSI(14)
    if "rsi" not in out.columns:
        delta = out["c"].diff()
        up = (delta.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean()
        dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = up / dn.replace(0, np.nan)
        out["rsi"] = 100 - (100/(1+rs))
    # EMAs
    if "ema20" not in out.columns: out["ema20"] = out["c"].ewm(span=20, adjust=False).mean()
    if "ema50" not in out.columns: out["ema50"] = out["c"].ewm(span=50, adjust=False).mean()
    # VWAP (intraday)
    if "vwap" not in out.columns:
        day = out.index.normalize()
        tp = (out["h"] + out["l"] + out["c"]) / 3.0
        pv = tp * out.get("v", 0).fillna(0)
        out["vwap"] = (pv.groupby(day).cumsum() / out.get("v", 0).fillna(0).groupby(day).cumsum().replace(0, np.nan)).ffill()
    return out

def load_cont(symbol: str, tf: str) -> pd.DataFrame:
    p = DATA_CONT / f"{symbol}_CONT_{tf}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_csv(p, parse_dates=["t"]).set_index("t").sort_index()

# ---- Simple long-only simulator with EOD flat ----
def eod_series(df: pd.DataFrame, hh: int, mm: int) -> pd.Series:
    m = (df.index.hour == hh) & (df.index.minute == mm)
    return pd.Series(m, index=df.index)

@dataclass
class Trade:
    t: pd.Timestamp
    side: str
    price: float
    ref: Optional[float]
    pnl: float
    equity: float

def simulate_long_only(
    df: pd.DataFrame,
    entry_mask: pd.Series,
    exit_rule,  # callable(hist_df)->bool for last bar
    cost_per_trade: float = 50.0,
    slip: float = 0.0,
    latest_entry: Optional[str] = None,
    eod_close: Tuple[int,int] = (15,25),
) -> pd.DataFrame:
    c = df["c"].astype(float)
    if latest_entry:
        hh, mm = map(int, latest_entry.split(":"))
        allowed = (df.index.hour < hh) | ((df.index.hour == hh) & (df.index.minute <= mm))
        entry_mask = entry_mask & allowed
    flatten = eod_series(df, *eod_close)

    in_pos = False
    ref = np.nan
    eq = 0.0
    trades: List[Trade] = []
    for t in df.index:
        px = float(c.at[t])
        if not in_pos:
            if bool(entry_mask.at[t]):
                ref = px + slip
                in_pos = True
                trades.append(Trade(t, "BUY", px, np.nan, 0.0, eq))
        else:
            # build rolling hist up to t
            hist = df.loc[:t]
            do_exit = bool(exit_rule(hist))
            if do_exit or bool(flatten.at[t]):
                fill = px - slip
                pnl = (fill - ref) - cost_per_trade
                eq += pnl
                trades.append(Trade(t, "EXIT_EOD" if flatten.at[t] else "SELL", px, ref, pnl, eq))
                in_pos = False
                ref = np.nan

    if in_pos:
        # force close at last bar
        t = df.index[-1]
        px = float(c.iloc[-1])
        pnl = (px - slip - ref) - cost_per_trade
        eq += pnl
        trades.append(Trade(t, "EXIT_EOD", px, ref, pnl, eq))

    return pd.DataFrame([tr.__dict__ for tr in trades]).set_index("t")

def summarize(tr: pd.DataFrame) -> Dict[str, float]:
    if tr.empty:
        return dict(trades=0, win_rate=0.0, total_pnl=0.0, avg_win=0.0, avg_loss=0.0, sharpe=0.0, mdd=0.0)
    pnl = tr.loc[tr["pnl"] != 0.0, "pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    daily = tr["equity"].groupby(tr.index.normalize()).last().diff().dropna()
    sharpe = (daily.mean() / (daily.std(ddof=0) + 1e-9)) * np.sqrt(252.0) if len(daily) else 0.0
    eq = tr["equity"].ffill().fillna(0)
    mdd = float((eq - eq.cummax()).min())
    return dict(
        trades=int(len(pnl)),
        win_rate=(len(wins) / max(1, len(pnl))) * 100.0,
        total_pnl=float(pnl.sum()),
        avg_win=float(wins.mean() if len(wins) else 0.0),
        avg_loss=float(losses.mean() if len(losses) else 0.0),
        sharpe=float(sharpe),
        mdd=mdd,
    )

# ---- RSI + trend filter strategy ----
def entry_mask_rsi(df: pd.DataFrame, entry_rsi: float, min_trend: float) -> pd.Series:
    return (df["rsi"] > entry_rsi) & ((df["ema20"] - df["ema50"]) > min_trend)

def exit_rule_rsi(exit_rsi: float):
    def _fn(hist: pd.DataFrame) -> bool:
        return bool(hist["rsi"].iloc[-1] < exit_rsi)
    return _fn

# ---- Walk-forward ----
@dataclass
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

def make_month_splits(
    df: pd.DataFrame,
    train_months: int = 3,
    test_months: int = 1,
) -> List[Split]:
    months = sorted(set((d.year, d.month) for d in df.index.date))
    splits: List[Split] = []
    for i in range(0, len(months) - (train_months + test_months) + 1):
        train_block = months[i : i + train_months]
        test_block  = months[i + train_months : i + train_months + test_months]
        t0 = pd.Timestamp(year=train_block[0][0], month=train_block[0][1], day=1)
        t1 = pd.Timestamp(year=train_block[-1][0], month=train_block[-1][1], day=28) + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59)
        s0 = pd.Timestamp(year=test_block[0][0], month=test_block[0][1], day=1)
        s1 = pd.Timestamp(year=test_block[-1][0], month=test_block[-1][1], day=28) + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59)
        splits.append(Split(t0, t1, s0, s1))
    return splits

def grid(params: Dict[str, List]) -> List[Dict]:
    keys = list(params.keys())
    from itertools import product
    out = []
    for vals in product(*[params[k] for k in keys]):
        out.append({k:v for k,v in zip(keys, vals)})
    return out

def run_fold(
    df: pd.DataFrame,
    split: Split,
    param_grid: List[Dict],
    cost: float,
    slip: float,
    latest_entry: str,
) -> Tuple[Dict, Dict]:
    train = df[(df.index >= split.train_start) & (df.index <= split.train_end)]
    test  = df[(df.index >= split.test_start) & (df.index <= split.test_end)]
    best_params, best_score = None, -1e18

    # Fit (select params by train total pnl)
    for p in param_grid:
        em = entry_mask_rsi(train, p["entry_rsi"], p["min_trend"])
        ex = exit_rule_rsi(p["exit_rsi"])
        tr = simulate_long_only(train, em, ex, cost_per_trade=cost, slip=slip, latest_entry=latest_entry)
        score = summarize(tr)["total_pnl"]
        if score > best_score:
            best_score, best_params = score, p

    # Evaluate on test
    em_t = entry_mask_rsi(test, best_params["entry_rsi"], best_params["min_trend"])
    ex_t = exit_rule_rsi(best_params["exit_rsi"])
    tr_t = simulate_long_only(test, em_t, ex_t, cost_per_trade=cost, slip=slip, latest_entry=latest_entry)
    s_t = summarize(tr_t)
    return best_params, s_t

def main():
    ap = argparse.ArgumentParser(description="Walk-forward RSI strategy (monthly splits).")
    ap.add_argument("--symbol", default="BANKNIFTY", choices=["BANKNIFTY","NIFTY"])
    ap.add_argument("--tf", default="5m", choices=["1m","3m","5m"])
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--train_months", type=int, default=3)
    ap.add_argument("--test_months", type=int, default=1)
    ap.add_argument("--cost", type=float, default=50.0)
    ap.add_argument("--slip", type=float, default=0.0)
    ap.add_argument("--latest_entry", default="15:00")
    # Param grid
    ap.add_argument("--entry_list", nargs="+", type=float, default=[60,62,64,66])
    ap.add_argument("--exit_list",  nargs="+", type=float, default=[50,52,54])
    ap.add_argument("--trend_list", nargs="+", type=float, default=[0,20,40])
    args = ap.parse_args()

    df = load_cont(args.symbol, args.tf)
    if args.start: df = df[df.index >= pd.Timestamp(args.start)]
    if args.end:   df = df[df.index <= pd.Timestamp(args.end)]
    df = compute_basic_features(df)

    splits = make_month_splits(df, train_months=args.train_months, test_months=args.test_months)
    if not splits:
        print("Not enough data for the chosen split sizes."); return

    pg = grid({"entry_rsi": args.entry_list, "exit_rsi": args.exit_list, "min_trend": args.trend_list})

    results = []
    for i, sp in enumerate(splits, 1):
        bp, st = run_fold(df, sp, pg, args.cost, args.slip, args.latest_entry)
        results.append((sp, bp, st))
        print(f"[Fold {i}] Train {sp.train_start.date()}→{sp.train_end.date()} | "
              f"Test {sp.test_start.date()}→{sp.test_end.date()} | "
              f"pick {bp} | OOS PnL {st['total_pnl']:.1f}, Sharpe {st['sharpe']:.2f}, Trades {st['trades']}")

    # Aggregate OOS
    oos_pnl = sum(r[2]["total_pnl"] for r in results)
    oos_trd = sum(r[2]["trades"] for r in results)
    oos_win = np.mean([r[2]["win_rate"] for r in results]) if results else 0.0
    print("\n=== Walk-Forward OOS Summary ===")
    print(f"Total OOS PnL: {oos_pnl:.2f}")
    print(f"Avg OOS WinRate: {oos_win:.2f}%")
    print(f"Total OOS Trades: {oos_trd}")

if __name__ == "__main__":
    main()
