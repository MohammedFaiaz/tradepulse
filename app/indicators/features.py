# app/indicators/features.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    roll_dn = pd.Series(dn, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_dn.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def _intraday_groups(idx: pd.DatetimeIndex) -> pd.Series:
    # Groups by date portion for intraday cumulations (e.g., VWAP)
    return pd.to_datetime(idx.date)

def _vwap(df: pd.DataFrame) -> pd.Series:
    # Null-safe: if 'v' missing/NaN, VWAP returns NaN for that row
    if "v" not in df.columns: 
        return pd.Series(index=df.index, dtype=float)
    tp = (df["h"] + df["l"] + df["c"]) / 3.0
    vol = df["v"].astype("float64")
    # Where volume is NaN -> keep VWAP NaN
    g = _intraday_groups(df.index)
    cum_pv = (tp * vol).groupby(g).cumsum()
    cum_v  = vol.groupby(g).cumsum()
    with np.errstate(divide='ignore', invalid='ignore'):
        vwap = cum_pv / cum_v
    # rows with NaN volume remain NaN
    vwap[cum_v == 0] = np.nan
    return vwap

def _cpr_from_prev_day(df: pd.DataFrame) -> pd.DataFrame:
    # Group intraday bars by *trading date* (no time)
    day_index = pd.to_datetime(df.index.date)

    # Daily H/L/C for each date
    daily = df.groupby(day_index).agg(
        ph=("h", "max"),
        pl=("l", "min"),
        pc=("c", "last"),
    )

    # Use the *previous day's* levels for today's CPR
    prev = daily.shift(1)

    # Compute CPR components on the daily frame
    pp_daily = (prev["ph"] + prev["pl"] + prev["pc"]) / 3.0
    tc_daily = (prev["ph"] + prev["pl"]) / 2.0
    bc_daily = 2 * pp_daily - tc_daily

    # Broadcast back to intraday rows via the date key
    # (reindex with the intraday day_index to align lengths)
    df = df.copy()
    df["pp"] = pp_daily.reindex(day_index).to_numpy()
    df["tc"] = tc_daily.reindex(day_index).to_numpy()
    df["bc"] = bc_daily.reindex(day_index).to_numpy()
    return df

def compute_features(df: pd.DataFrame, add_vwap: bool = True) -> pd.DataFrame:
    """
    Input df: index datetime; columns at least: o,h,l,c[,v,oi][,data_source]
    Returns df with added: rsi, atr, ema20, ema50, (optional) vwap, pp, tc, bc, has_volume
    """
    out = df.copy()
    out["rsi"] = _rsi(out["c"])
    out["atr"] = _atr(out["h"], out["l"], out["c"])
    out["ema20"] = _ema(out["c"], 20)
    out["ema50"] = _ema(out["c"], 50)

    if add_vwap:
        out["vwap"] = _vwap(out)

    out = _cpr_from_prev_day(out)
    out["has_volume"] = out["v"].notna()
    return out
