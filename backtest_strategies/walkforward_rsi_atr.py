# backtest_strategies/walkforward_rsi_atr.py
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# ---- Project paths ----
DATA_CONT = Path(r"C:\tradePulse\data\continuous")

# ---- Holidays / expiry skip (extend as needed) ----
HOLIDAYS = set([
    # '2025-10-28', '2025-11-04',  # example
])
def is_holiday(d: pd.Timestamp) -> bool:
    return d.strftime("%Y-%m-%d") in HOLIDAYS

# ---- Features ----
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["o","h","l","c"]:
        if col not in out.columns:
            raise ValueError("Data must include o,h,l,c columns.")
    for col in ["v","oi"]:
        if col not in out.columns:
            out[col] = np.nan

    # RSI(14)
    delta = out["c"].diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out["rsi"] = 100 - (100/(1+rs))

    # EMAs
    out["ema20"] = out["c"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["c"].ewm(span=50, adjust=False).mean()

    # ATR(14)
    tr = pd.concat([
        (out["h"] - out["l"]),
        (out["h"] - out["c"].shift()).abs(),
        (out["l"] - out["c"].shift()).abs()
    ], axis=1).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # VWAP (intraday)
    day = out.index.normalize()
    tp = (out["h"] + out["l"] + out["c"]) / 3.0
    vol = out.get("v", 0).fillna(0)
    pv = (tp * vol).groupby(day).cumsum()
    vv = vol.groupby(day).cumsum().replace(0, np.nan)
    out["vwap"] = (pv / vv).ffill()

    return out

def load_cont(symbol: str, tf: str) -> pd.DataFrame:
    p = DATA_CONT / f"{symbol}_CONT_{tf}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_csv(p, parse_dates=["t"]).set_index("t").sort_index()

# ---- Strategy masks ----
def entry_mask_rsi(df: pd.DataFrame, entry_rsi: float, min_trend: float) -> pd.Series:
    return (df["rsi"] > entry_rsi) & ((df["ema20"] - df["ema50"]) > min_trend)

def entry_mask_rsi_short(df: pd.DataFrame, entry_rsi_short: float, min_trend_short: float) -> pd.Series:
    return (df["rsi"] < entry_rsi_short) & ((df["ema20"] - df["ema50"]) < -min_trend_short)

# ---- Utilities ----
def within_latest_entry(ts: pd.Timestamp, latest: Optional[str]) -> bool:
    if not latest:
        return True
    hh, mm = map(int, latest.split(":"))
    return (ts.hour < hh) or (ts.hour == hh and ts.minute <= mm)

def session_filter(index: pd.DatetimeIndex, skip_first_min: int, skip_last_min: int) -> pd.Series:
    # Keep bars that are NOT in the skipped zones
    day = index.normalize()
    first = index.groupby(day).transform("min")
    last  = index.groupby(day).transform("max")
    ok_start = index >= (first + pd.Timedelta(minutes=skip_first_min))
    ok_end   = index <= (last  - pd.Timedelta(minutes=skip_last_min))
    return ok_start & ok_end

def one_trade_per_day_guard(dates_hit: Dict[pd.Timestamp, bool], t: pd.Timestamp, enabled: bool) -> bool:
    if not enabled: return True
    d = t.normalize()
    return not dates_hit.get(d, False)

# ---- Simulator (long/short, ATR stops & targets, cooldown, one-per-day) ----
@dataclass
class Trade:
    t: pd.Timestamp
    side: str
    price: float
    ref: Optional[float]
    pnl: float
    equity: float

def simulate(
    df: pd.DataFrame,
    long_mask: pd.Series,
    short_mask: Optional[pd.Series],
    cost_per_trade: float,
    slip: float,
    atr_stop_k: Optional[float],
    atr_target_k: Optional[float],
    latest_entry: Optional[str],
    skip_first_min: int,
    skip_last_min: int,
    cooldown_bars: int,
    one_per_day: bool,
    allow_short: bool,
    eod_close: Tuple[int,int] = (15,25),
) -> pd.DataFrame:

    idx = df.index
    c = df["c"].astype(float)
    atr = df["atr14"].astype(float)

    # bars allowed by session filter
    sess_ok = session_filter(idx, skip_first_min, skip_last_min)

    # EOD flatten series (one bar before 15:30 default for 5m)
    eod = (idx.hour == eod_close[0]) & (idx.minute == eod_close[1])

    eq = 0.0
    trades: List[Trade] = []

    pos = None            # None | "LONG" | "SHORT"
    ref = np.nan          # entry reference price
    stop_px = np.nan
    tgt_px = np.nan
    cool = 0              # bars remaining in cooldown
    day_hit: Dict[pd.Timestamp, bool] = {}  # one-per-day tracker

    for i, t in enumerate(idx):
        px = float(c.iat[i])
        if cool > 0:
            cool -= 1

        # EOD flatten if in position
        if pos is not None and eod[i]:
            pnl = ((px - slip) - ref if pos == "LONG" else (ref - (px + slip))) - cost_per_trade
            eq += pnl
            trades.append(Trade(t, "EXIT_EOD", px, ref, pnl, eq))
            pos, ref, stop_px, tgt_px = None, np.nan, np.nan, np.nan
            day_hit[t.normalize()] = True
            cool = cooldown_bars
            continue

        # If not in session window, do nothing (but stops can still apply if in pos)
        if not sess_ok.iat[i]:
            # protective: if stop/target hit outside session window
            if pos is not None:
                if (atr_stop_k is not None and ((px <= stop_px and pos=="LONG") or (px >= stop_px and pos=="SHORT"))) \
                   or (atr_target_k is not None and ((px >= tgt_px and pos=="LONG") or (px <= tgt_px and pos=="SHORT"))):
                    pnl = ((px - slip) - ref if pos == "LONG" else (ref - (px + slip))) - cost_per_trade
                    eq += pnl
                    trades.append(Trade(t, "EXIT", px, ref, pnl, eq))
                    pos, ref, stop_px, tgt_px = None, np.nan, np.nan, np.nan
                    day_hit[t.normalize()] = True
                    cool = cooldown_bars
            continue

        # Manage open position: stops/targets first
        if pos is not None:
            hit_stop = (atr_stop_k is not None) and ((px <= stop_px and pos=="LONG") or (px >= stop_px and pos=="SHORT"))
            hit_tgt  = (atr_target_k is not None) and ((px >= tgt_px and pos=="LONG") or (px <= tgt_px and pos=="SHORT"))
            if hit_stop or hit_tgt:
                pnl = ((px - slip) - ref if pos == "LONG" else (ref - (px + slip))) - cost_per_trade
                eq += pnl
                trades.append(Trade(t, "EXIT", px, ref, pnl, eq))
                pos, ref, stop_px, tgt_px = None, np.nan, np.nan, np.nan
                day_hit[t.normalize()] = True
                cool = cooldown_bars
                continue
            else:
                # optional: trailing target/stop could be added here
                pass

        # Flat: consider entries
        if pos is None and cool == 0 and one_trade_per_day_guard(day_hit, t, one_per_day) and within_latest_entry(t, latest_entry):
            go_long  = bool(long_mask.iat[i])
            go_short = bool(short_mask.iat[i]) if (allow_short and short_mask is not None) else False

            if go_long and not go_short:
                pos = "LONG"
                ref = px + slip
                a = float(atr.iat[i]) if not np.isnan(atr.iat[i]) else 0.0
                stop_px = ref - (atr_stop_k * a) if atr_stop_k is not None else -np.inf
                tgt_px  = ref + (atr_target_k * a) if atr_target_k is not None else np.inf
                trades.append(Trade(t, "BUY", px, np.nan, 0.0, eq))
            elif go_short:
                pos = "SHORT"
                ref = px - slip
                a = float(atr.iat[i]) if not np.isnan(atr.iat[i]) else 0.0
                stop_px = ref + (atr_stop_k * a) if atr_stop_k is not None else np.inf
                tgt_px  = ref - (atr_target_k * a) if atr_target_k is not None else -np.inf
                trades.append(Trade(t, "SHORT", px, np.nan, 0.0, eq))

    # Force close at file end
    if pos is not None:
        t = df.index[-1]
        px = float(c.iloc[-1])
        pnl = ((px - slip) - ref if pos == "LONG" else (ref - (px + slip))) - cost_per_trade
        eq += pnl
        trades.append(Trade(t, "EXIT_EOD", px, ref, pnl, eq))

    return pd.DataFrame([tr.__dict__ for tr in trades]).set_index("t")

# ---- Metrics ----
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

# ---- Walk-forward machinery ----
@dataclass
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

def make_month_splits(df: pd.DataFrame, train_months: int, test_months: int) -> List[Split]:
    months = sorted(set((d.year, d.month) for d in df.index.date))
    res = []
    for i in range(0, len(months) - (train_months + test_months) + 1):
        tr = months[i : i + train_months]
        te = months[i + train_months : i + train_months + test_months]
        t0 = pd.Timestamp(tr[0][0], tr[0][1], 1)
        t1 = (pd.Timestamp(tr[-1][0], tr[-1][1], 28) + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59))
        s0 = pd.Timestamp(te[0][0], te[0][1], 1)
        s1 = (pd.Timestamp(te[-1][0], te[-1][1], 28) + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59))
        res.append(Split(t0, t1, s0, s1))
    return res

def grid(params: Dict[str, List]) -> List[Dict]:
    from itertools import product
    keys = list(params.keys())
    return [{k:v for k,v in zip(keys, vals)} for vals in product(*[params[k] for k in keys])]

def run_fold(
    df: pd.DataFrame,
    sp: Split,
    param_grid: List[Dict],
    sim_args: Dict,
    allow_short: bool,
) -> Tuple[Dict, Dict, pd.DataFrame]:
    train = df[(df.index >= sp.train_start) & (df.index <= sp.train_end)]
    test  = df[(df.index >= sp.test_start) & (df.index <= sp.test_end)]

    # Remove holidays/expiry if you populate HOLIDAYS
    train = train[~train.index.normalize().isin({pd.Timestamp(h) for h in HOLIDAYS})]
    test  = test[~test.index.normalize().isin({pd.Timestamp(h) for h in HOLIDAYS})]

    best_params, best_score = None, -1e18

    for p in param_grid:
        long_m  = entry_mask_rsi(train, p["entry_rsi"], p["min_trend"])
        short_m = entry_mask_rsi_short(train, p["entry_rsi_short"], p["min_trend_short"]) if allow_short else None
        trd = simulate(train, long_m, short_m, **sim_args, allow_short=allow_short)
        sc = summarize(trd)["total_pnl"]
        if sc > best_score:
            best_score, best_params = sc, p

    # OOS
    long_m  = entry_mask_rsi(test, best_params["entry_rsi"], best_params["min_trend"])
    short_m = entry_mask_rsi_short(test, best_params["entry_rsi_short"], best_params["min_trend_short"]) if allow_short else None
    tr_out  = simulate(test, long_m, short_m, **sim_args, allow_short=allow_short)
    s_out   = summarize(tr_out)
    return best_params, s_out, tr_out

def main():
    ap = argparse.ArgumentParser(description="Walk-forward RSI w/ ATR stops/targets, long/short, session filters.")
    ap.add_argument("--symbol", default="BANKNIFTY", choices=["BANKNIFTY","NIFTY"])
    ap.add_argument("--tf", default="5m", choices=["1m","3m","5m"])
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--train_months", type=int, default=3)
    ap.add_argument("--test_months", type=int, default=1)

    # Costs and execution
    ap.add_argument("--cost", type=float, default=50.0)
    ap.add_argument("--slip", type=float, default=0.0)

    # Risk model
    ap.add_argument("--atr_stop_k", type=float, default=0.7)
    ap.add_argument("--atr_target_k", type=float, default=1.2)

    # Session controls
    ap.add_argument("--latest_entry", default="15:00")
    ap.add_argument("--skip_first_min", type=int, default=5)
    ap.add_argument("--skip_last_min", type=int, default=5)

    # Trade flow
    ap.add_argument("--cooldown", type=int, default=5)
    ap.add_argument("--one_per_day", action="store_true")
    ap.add_argument("--allow_short", action="store_true")

    # Param grid
    ap.add_argument("--entry_list", nargs="+", type=float, default=[60,62,64,66])
    ap.add_argument("--exit_list",  nargs="+", type=float, default=[50,52,54])  # used to gate exits via RSI cross below; we keep ATR exits primary
    ap.add_argument("--trend_list", nargs="+", type=float, default=[0,20,40])
    ap.add_argument("--entry_short_list", nargs="+", type=float, default=[40,38,36])  # RSI below
    ap.add_argument("--trend_short_list", nargs="+", type=float, default=[0,20,40])

    ap.add_argument("--oos_csv", default="walkforward_oos_detail.csv")

    args = ap.parse_args()

    df = load_cont(args.symbol, args.tf)
    if args.start: df = df[df.index >= pd.Timestamp(args.start)]
    if args.end:   df = df[df.index <= pd.Timestamp(args.end)]
    df = compute_features(df)

    splits = make_month_splits(df, args.train_months, args.test_months)
    if not splits:
        print("Not enough data for chosen train/test split.")
        return

    pg = grid({
        "entry_rsi": args.entry_list,
        "exit_rsi":  args.exit_list,
        "min_trend": args.trend_list,
        "entry_rsi_short": args.entry_short_list,
        "min_trend_short": args.trend_short_list,
    })

    sim_args = dict(
        cost_per_trade=args.cost,
        slip=args.slip,
        atr_stop_k=args.atr_stop_k,
        atr_target_k=args.atr_target_k,
        latest_entry=args.latest_entry,
        skip_first_min=args.skip_first_min,
        skip_last_min=args.skip_last_min,
        cooldown_bars=args.cooldown,
        one_per_day=args.one_per_day,
        eod_close=(15, 25 if args.tf != "1m" else 29),
    )

    all_rows = []
    oos_pnl = 0.0
    oos_trd = 0
    oos_wr  = []

    for i, sp in enumerate(splits, 1):
        bp, st, trd = run_fold(df, sp, pg, sim_args, allow_short=args.allow_short)
        oos_pnl += st["total_pnl"]
        oos_trd += st["trades"]
        oos_wr.append(st["win_rate"])
        print(f"[Fold {i}] Train {sp.train_start.date()}→{sp.train_end.date()} | "
              f"Test {sp.test_start.date()}→{sp.test_end.date()} | "
              f"pick {bp} | OOS PnL {st['total_pnl']:.1f}, Sharpe {st['sharpe']:.2f}, Trades {st['trades']}")

        # annotate and collect trades detail
        tmp = trd.copy()
        tmp["fold"] = i
        tmp["train_start"] = sp.train_start
        tmp["train_end"]   = sp.train_end
        tmp["test_start"]  = sp.test_start
        tmp["test_end"]    = sp.test_end
        tmp["params"]      = str(bp)
        all_rows.append(tmp)

    print("\n=== Walk-Forward OOS Summary ===")
    print(f"Total OOS PnL: {oos_pnl:.2f}")
    print(f"Avg OOS WinRate: {np.mean(oos_wr) if oos_wr else 0.0:.2f}%")
    print(f"Total OOS Trades: {oos_trd}")

    if all_rows:
        out = pd.concat(all_rows).sort_index()
        out.to_csv(args.oos_csv, index=True, date_format="%Y-%m-%d %H:%M:%S")
        print(f"Saved OOS trade log -> {args.oos_csv}")

if __name__ == "__main__":
    main()
