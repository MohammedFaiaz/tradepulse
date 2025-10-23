# quick_backtest.py (long & short, ATR stops/targets, guards)
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Callable, Optional, Literal

import numpy as np
import pandas as pd

# --- CONFIG PATHS (match your project) ---
DATA_CONT = Path(r"C:\tradePulse\data\continuous")

# ----------------- Feature engineering -----------------
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["h"].astype(float), df["l"].astype(float), df["c"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_index().copy()

    # Ensure presence of optional cols (non-blocking)
    for col in ["v", "oi", "series", "data_source"]:
        if col not in out.columns:
            out[col] = np.nan

    # RSI (14)
    if "rsi" not in out.columns:
        delta = out["c"].diff()
        up = (delta.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = up / down.replace(0, np.nan)
        out["rsi"] = 100 - (100 / (1 + rs))

    # EMAs
    if "ema20" not in out.columns:
        out["ema20"] = _ema(out["c"], 20)
    if "ema50" not in out.columns:
        out["ema50"] = _ema(out["c"], 50)

    # VWAP (intraday reset)
    if "vwap" not in out.columns:
        day = out.index.normalize()
        # typical price
        tp = (out["h"] + out["l"] + out["c"]) / 3.0
        vol = out.get("v", 0).fillna(0)
        cum_pv = (tp * vol).groupby(day).cumsum()
        cum_v = vol.groupby(day).cumsum().replace(0, np.nan)
        out["vwap"] = (cum_pv / cum_v).ffill()

    # CPR (previous day pivot)
    if not {"pp", "tc", "bc"}.issubset(out.columns):
        d = out.index.normalize()
        daily_h = out["h"].groupby(d).transform("max").shift(1)
        daily_l = out["l"].groupby(d).transform("min").shift(1)
        daily_c = out["c"].groupby(d).transform("last").shift(1)
        pp = (daily_h + daily_l + daily_c) / 3.0
        bc = (daily_h + daily_l) / 2.0
        tc = 2 * pp - bc
        out["pp"], out["bc"], out["tc"] = pp, bc, tc

    # ATR14
    if "atr" not in out.columns:
        out["atr"] = _atr(out, 14)

    # Drop initial warmup NaNs for RSI/EMA/ATR (keep CPR/VWAP as is)
    out = out.dropna(subset=["rsi", "ema20", "ema50", "atr"], how="any")
    return out

# ----------------- Data loader -----------------
def load_cont(symbol: str, tf: str) -> pd.DataFrame:
    p = DATA_CONT / f"{symbol}_CONT_{tf}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p, parse_dates=["t"]).set_index("t").sort_index()
    needed = {"o", "h", "l", "c"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{p} must have columns {needed}; got {df.columns.tolist()}")
    return df

# ----------------- Helpers -----------------
def eod_flat_mask(df: pd.DataFrame, minute_close: Tuple[int, int] = (15, 25)) -> pd.Series:
    """Boolean *Series* that’s True on the bar where we should flatten before 15:30."""
    mask = ((df.index.hour == minute_close[0]) &
            (df.index.minute == minute_close[1]))
    return pd.Series(mask, index=df.index)


def within_latest_entry(df: pd.DataFrame, latest_hhmm: Optional[str]) -> pd.Series:
    if not latest_hhmm:
        return pd.Series(True, index=df.index)
    hh, mm = map(int, latest_hhmm.split(":"))
    tmask = (df.index.hour < hh) | ((df.index.hour == hh) & (df.index.minute <= mm))
    return pd.Series(tmask, index=df.index)

# ----------------- Trade model -----------------
@dataclass
class TradeRow:
    t: pd.Timestamp
    side: str      # BUY/SELL/SHORT/COVER/EXIT_EOD
    price: float
    ref: Optional[float]
    pnl: float
    equity: float

# ----------------- Strategies (signals only) -----------------
def strat_rsi_pullback_signals(
    df: pd.DataFrame,
    entry_rsi: float = 60,
    exit_rsi: float = 50,
    min_trend: float = 0.0
) -> Tuple[pd.Series, pd.Series, Callable[[pd.Series, pd.Series, pd.Timestamp, Literal["long","short"]], bool]]:
    """
    Returns (long_entry, short_entry, exit_checker)
    exit_checker(hist_c, hist_rsi, now_t, side) -> bool
    """
    rsi = df["rsi"]; ema20 = df["ema20"]; ema50 = df["ema50"]

    long_entry = (rsi > entry_rsi) & ((ema20 - ema50) > min_trend)
    short_entry = (rsi < (100 - entry_rsi)) & ((ema50 - ema20) > min_trend)

    def exit_checker(hist_c: pd.Series, hist_rsi: pd.Series, now_t: pd.Timestamp, side: Literal["long","short"]) -> bool:
        r = float(hist_rsi.iloc[-1])
        if side == "long":
            return r < exit_rsi
        else:
            # symmetric: cover when rsi > (100 - exit_rsi)
            return r > (100 - exit_rsi)

    return long_entry, short_entry, exit_checker

def strat_orb_signals(
    df: pd.DataFrame,
    minutes_open: int = 15,
    buffer_pts: float = 0.0
):
    d = df.copy()
    step_min = int((d.index[1] - d.index[0]).total_seconds() // 60)
    bars = max(1, int(minutes_open / max(1, step_min)))
    day = d.index.normalize()
    is_open = d.groupby(day).cumcount() < bars
    orh = d["h"].where(is_open).groupby(day).transform("max")
    orl = d["l"].where(is_open).groupby(day).transform("min")

    long_entry = d["c"] > (orh + buffer_pts)
    short_entry = d["c"] < (orl - buffer_pts)

    def exit_checker(hist_c: pd.Series, hist_rsi: pd.Series, now_t: pd.Timestamp, side: Literal["long","short"]) -> bool:
        px = float(hist_c.iloc[-1])
        if side == "long":
            return px < orh.loc[now_t]
        else:
            return px > orl.loc[now_t]

    return long_entry, short_entry, exit_checker

def strat_trend_vwap_signals(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
    vwap_confirm: bool = True,
    cpr_mid_confirm: bool = False
):
    ef = _ema(df["c"], ema_fast)
    es = _ema(df["c"], ema_slow)
    above_ema = ef > es
    below_ema = ef < es
    above_vwap = df["c"] > df["vwap"]
    below_vwap = df["c"] < df["vwap"]
    above_cpr_mid = df["c"] > df["pp"]
    below_cpr_mid = df["c"] < df["pp"]

    long_entry = above_ema & (above_vwap if vwap_confirm else True) & (above_cpr_mid if cpr_mid_confirm else True)
    short_entry = below_ema & (below_vwap if vwap_confirm else True) & (below_cpr_mid if cpr_mid_confirm else True)

    def exit_checker(hist_c: pd.Series, hist_rsi: pd.Series, now_t: pd.Timestamp, side: Literal["long","short"]) -> bool:
        ef_now = _ema(hist_c, ema_fast).iloc[-1]
        vw_now = hist_c.iloc[-1] * 0 + df["vwap"].reindex(hist_c.index).ffill().iloc[-1]
        px = float(hist_c.iloc[-1])
        if side == "long":
            return (px < ef_now) and (px < vw_now)
        else:
            return (px > ef_now) and (px > vw_now)

    return long_entry, short_entry, exit_checker

# ----------------- Simulator -----------------
def simulate_long_short(
    df: pd.DataFrame,
    long_entry: pd.Series,
    short_entry: pd.Series,
    exit_checker: Callable[[pd.Series, pd.Series, pd.Timestamp, Literal["long","short"]], bool],
    price_col: str = "c",
    cost_per_round: float = 50.0,
    slippage_pts: float = 0.0,
    eod_close: Tuple[int, int] = (15, 25),
    latest_entry: Optional[str] = None,
    min_hold_bars: int = 0,
    cooldown_bars: int = 0,
    one_per_day: bool = False,
    atr_stop_k: Optional[float] = None,
    atr_target_k: Optional[float] = None,
) -> pd.DataFrame:

    c = df[price_col].astype(float)
    rsi = df["rsi"].astype(float)
    atr = df["atr"].astype(float)

    long_entry = long_entry.reindex(df.index).fillna(False)
    short_entry = short_entry.reindex(df.index).fillna(False)

    eod = eod_flat_mask(df, eod_close)
    le_mask = within_latest_entry(df, latest_entry)

    # State
    pos: Optional[Literal["long","short"]] = None
    ref: float = np.nan
    eq: float = 0.0
    entry_t: Optional[pd.Timestamp] = None
    stop_lvl: Optional[float] = None
    tgt_lvl: Optional[float] = None
    cool_left = 0
    traded_today: Dict[pd.Timestamp, bool] = {}

    rows: list[TradeRow] = []

    # Precompute day index for one_per_day
    day = df.index.normalize()

    for i, t in enumerate(df.index):
        px = float(c.iat[i])
        rs = float(rsi.iat[i])

        # Reset "traded_today" per new day
        dkey = day[i]
        if dkey not in traded_today:
            traded_today[dkey] = False

        # cooldown ticks
        if cool_left > 0:
            cool_left -= 1

        # EOD exit if in position
        if pos is not None and eod.iat[i]:
            # close
            fill = px - slippage_pts if pos == "long" else px + slippage_pts
            pnl = (fill - ref) if pos == "long" else (ref - fill)
            pnl -= cost_per_round
            eq += pnl
            rows.append(TradeRow(t, "EXIT_EOD", px, ref, pnl, eq))
            pos, ref, entry_t, stop_lvl, tgt_lvl = None, np.nan, None, None, None
            continue

        # If flat: consider entries (respect latest_entry, cooldown, one_per_day)
        if pos is None:
            if not le_mask.iat[i]:
                continue
            if cool_left > 0:
                continue
            if one_per_day and traded_today.get(dkey, False):
                continue

            # choose side if any
            enter_long = bool(long_entry.iat[i])
            enter_short = bool(short_entry.iat[i])

            # if both true, prefer the stronger impulse (distance of rsi from mid)
            side_to_take: Optional[Literal["long","short"]] = None
            if enter_long and not enter_short:
                side_to_take = "long"
            elif enter_short and not enter_long:
                side_to_take = "short"
            elif enter_long and enter_short:
                side_to_take = "long" if (rs - 50) >= (50 - rs) else "short"

            if side_to_take:
                pos = side_to_take
                traded_today[dkey] = True
                entry_t = t
                if pos == "long":
                    ref = px + slippage_pts
                    rows.append(TradeRow(t, "BUY", px, np.nan, 0.0, eq))
                else:
                    ref = px - slippage_pts
                    rows.append(TradeRow(t, "SHORT", px, np.nan, 0.0, eq))

                # Fix ATR stop/target at entry
                if atr_stop_k is not None:
                    stop_lvl = (ref - atr_stop_k * float(atr.iat[i])) if pos == "long" else (ref + atr_stop_k * float(atr.iat[i]))
                if atr_target_k is not None:
                    tgt_lvl = (ref + atr_target_k * float(atr.iat[i])) if pos == "long" else (ref - atr_target_k * float(atr.iat[i]))
                continue

        # If in position: check exits (min hold, stop/target, strategy exit)
        if pos is not None:
            held = (i - df.index.get_indexer([entry_t])[0]) if entry_t is not None else 0
            can_exit = held >= max(0, min_hold_bars)

            do_exit = False
            reason = "RULE"

            # hard stops/targets first
            if atr_stop_k is not None and stop_lvl is not None:
                if (pos == "long" and px <= stop_lvl) or (pos == "short" and px >= stop_lvl):
                    do_exit = True
                    reason = "STOP"
            if not do_exit and atr_target_k is not None and tgt_lvl is not None:
                if (pos == "long" and px >= tgt_lvl) or (pos == "short" and px <= tgt_lvl):
                    do_exit = True
                    reason = "TARGET"

            # strategy exit
            if not do_exit and can_exit:
                hist_c = c.loc[:t]
                hist_r = rsi.loc[:t]
                if exit_checker(hist_c, hist_r, t, pos):
                    do_exit = True
                    reason = "STRAT"

            if do_exit:
                fill = px - slippage_pts if pos == "long" else px + slippage_pts
                pnl = (fill - ref) if pos == "long" else (ref - fill)
                pnl -= cost_per_round
                eq += pnl
                rows.append(TradeRow(t, "SELL" if pos == "long" else "COVER", px, ref, pnl, eq))
                pos, ref, entry_t, stop_lvl, tgt_lvl = None, np.nan, None, None, None
                cool_left = max(cool_left, cooldown_bars)
                continue

    # Flatten if still holding at file end
    if pos is not None:
        t = df.index[-1]
        px = float(c.iat[-1])
        fill = px - slippage_pts if pos == "long" else px + slippage_pts
        pnl = (fill - ref) if pos == "long" else (ref - fill)
        pnl -= cost_per_round
        eq += pnl
        rows.append(TradeRow(t, "EXIT_EOD", px, ref, pnl, eq))

    out = pd.DataFrame([r.__dict__ for r in rows]).set_index("t")
    return out

# ----------------- Metrics -----------------
def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return dict(trades=0, win_rate=0.0, total_pnl=0.0, avg_win=0.0, avg_loss=0.0,
                    sharpe_daily252=0.0, max_drawdown=0.0)
    pnl = trades.loc[trades["pnl"] != 0.0, "pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    eq = trades["equity"].ffill().fillna(0)
    daily = eq.groupby(eq.index.normalize()).last().diff().dropna()
    sharpe = 0.0 if daily.std(ddof=0) == 0 else (daily.mean() / daily.std(ddof=0)) * np.sqrt(252.0)
    roll_max = eq.cummax()
    dd = (eq - roll_max)
    return dict(
        trades=int(len(pnl)),
        win_rate=(len(wins) / max(1, len(pnl))) * 100.0,
        total_pnl=float(pnl.sum()),
        avg_win=float(wins.mean() if len(wins) else 0.0),
        avg_loss=float(losses.mean() if len(losses) else 0.0),
        sharpe_daily252=float(sharpe),
        max_drawdown=float(dd.min() if len(dd) else 0.0),
    )

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="Quick pluggable backtester (intraday flat).")
    ap.add_argument("--symbol", default="BANKNIFTY", choices=["BANKNIFTY", "NIFTY"])
    ap.add_argument("--tf", default="5m", choices=["1m", "3m", "5m"])
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--strategy", default="rsi_pullback", choices=["rsi_pullback", "orb", "trend_vwap"])
    ap.add_argument("--cost", type=float, default=50.0)
    ap.add_argument("--slip", type=float, default=0.0)

    # Strategy params
    ap.add_argument("--entry_rsi", type=float, default=62.0)
    ap.add_argument("--exit_rsi",  type=float, default=52.0)
    ap.add_argument("--min_trend", type=float, default=0.0)
    ap.add_argument("--orb_minutes", type=int, default=15)
    ap.add_argument("--orb_buffer", type=float, default=0.0)
    ap.add_argument("--ema_fast", type=int, default=20)
    ap.add_argument("--ema_slow", type=int, default=50)
    ap.add_argument("--vwap_confirm", action="store_true")
    ap.add_argument("--cpr_confirm", action="store_true")

    # Position controls
    ap.add_argument("--enable_short", action="store_true")
    ap.add_argument("--latest_entry", default=None, help="HH:MM (local) cutoff for new entries")
    ap.add_argument("--min_hold", type=int, default=0, help="min bars to hold before rule exit allowed")
    ap.add_argument("--cooldown", type=int, default=0, help="bars to wait after an exit")
    ap.add_argument("--one_per_day", action="store_true")

    # ATR-based exits sized at entry
    ap.add_argument("--atr_stop_k", type=float, default=None, help="e.g. 0.6 -> stop = entry -/+ 0.6*ATR")
    ap.add_argument("--atr_target_k", type=float, default=None, help="e.g. 1.6 -> target = entry +/- 1.6*ATR")

    args = ap.parse_args()

    df = load_cont(args.symbol, args.tf)
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end)]
    df = compute_basic_features(df)

    # Build signals
    if args.strategy == "rsi_pullback":
        long_e, short_e, exit_checker = strat_rsi_pullback_signals(
            df, entry_rsi=args.entry_rsi, exit_rsi=args.exit_rsi, min_trend=args.min_trend
        )
    elif args.strategy == "orb":
        long_e, short_e, exit_checker = strat_orb_signals(
            df, minutes_open=args.orb_minutes, buffer_pts=args.orb_buffer
        )
    else:
        long_e, short_e, exit_checker = strat_trend_vwap_signals(
            df, ema_fast=args.ema_fast, ema_slow=args.ema_slow,
            vwap_confirm=args.vwap_confirm, cpr_mid_confirm=args.cpr_confirm
        )

    if not args.enable_short:
        short_e = pd.Series(False, index=df.index)

    trades = simulate_long_short(
        df,
        long_entry=long_e,
        short_entry=short_e,
        exit_checker=exit_checker,
        cost_per_round=args.cost,
        slippage_pts=args.slip,
        eod_close=(15, 25 if args.tf != "1m" else 29),
        latest_entry=args.latest_entry,
        min_hold_bars=args.min_hold,
        cooldown_bars=args.cooldown,
        one_per_day=args.one_per_day,
        atr_stop_k=args.atr_stop_k,
        atr_target_k=args.atr_target_k,
    )

    s = summarize_trades(trades)
    print("\n=== Backtest Summary ===")
    for k, v in s.items():
        print(f"{k:>16}: {v}")
    print("\nLast 10 trade rows:")
    print(trades.tail(10))

if __name__ == "__main__":
    main()
