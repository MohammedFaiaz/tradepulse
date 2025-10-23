# ingest_csv_to_1m.py
# Reads your tick CSVs (I/II/III for NIFTY & BANKNIFTY), builds 1-minute OHLCV+OI,
# writes per-day files, and produces a front-month selection map (by daily volume).
#
# Input layout examples:
# C:\tradePulse\History_tickData\2024\AUG_2024\GFDLNFO_TICK_01082024\GFDLNFO_TICK_01082024\BANKNIFTY-I.NFO.CSV
# C:\tradePulse\History_tickData\2025\APR_2025\GFDLNFO_TICK_01042025\GFDLNFO_TICK_01042025\NIFTY-III.NFO.CSV

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Tuple
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==== CONFIG ====
INPUT_ROOTS = [
    r"C:\tradePulse\History_tickData\2024",
    r"C:\tradePulse\History_tickData\2025",
]
OUTPUT_ROOT = Path(r"C:\tradePulse\data")
SYMBOLS = ("BANKNIFTY", "NIFTY")
SERIES = ("I", "II", "III")  # current, next, far
# Trading session (optional filter); leave None to keep full day
SESSION_START = "09:15:00"
SESSION_END   = "15:30:00"
# ==============

def parse_name(file: Path) -> Tuple[str, str]:
    """
    BANKNIFTY-I.NFO.CSV -> ("BANKNIFTY", "I")
    NIFTY-III.NFO.CSV   -> ("NIFTY", "III")
    """
    stem = file.name.upper()
    # e.g., BANKNIFTY-I.NFO.CSV -> BANKNIFTY-I
    base = stem.split(".")[0]
    # e.g., BANKNIFTY-I -> ["BANKNIFTY", "I"]
    parts = base.split("-")
    if len(parts) != 2:
        raise ValueError(f"Unexpected file name: {file.name}")
    return parts[0], parts[1]

def find_tick_files() -> list[Path]:
    files: list[Path] = []
    for root in INPUT_ROOTS:
        root_p = Path(root)
        if not root_p.exists():
            continue
        # recursively find *.CSV under *\GFDLNFO_TICK_*\GFDLNFO_TICK_*\*.CSV
        for p in root_p.rglob("GFDLNFO_TICK_*"):
            if p.is_dir():
                files.extend([f for f in p.glob("*.CSV") if f.is_file()])
    # Keep only BANKNIFTY/NIFTY and I/II/III
    out = []
    for f in files:
        try:
            sym, ser = parse_name(f)
            if sym in SYMBOLS and ser in SERIES:
                out.append(f)
        except Exception:
            pass
    return sorted(out)

def load_and_aggregate_1m(file: Path) -> tuple[pd.DataFrame, str, str, str]:
    """
    Returns (df_1m, symbol, series, ymd)
    df_1m index=minute(datetime), columns=[o,h,l,c,v,oi]
    """
    symbol, series = parse_name(file)

    # Derive ymd (YYYY-MM-DD) from the parent folder name (e.g., ..._01082024)
    # and/or from CSV. We'll compute from CSV safely.
    usecols = ["Ticker", "Date", "Time", "LTP", "LTQ", "OpenInterest"]
    df = pd.read_csv(
        file,
        usecols=usecols,
        dtype={
            "Ticker": "string",
            "Date": "string",
            "Time": "string",
            "LTP": "float64",
            "LTQ": "int64",
            "OpenInterest": "Int64"
        }
    )

    # Combine Date + Time -> datetime (IST local; we keep naive timestamps in IST)
    # Sample date looks like "01/08/2024" (DD/MM/YYYY)
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df = df.assign(datetime=dt)
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")

    # Optional: restrict to trading session
    if SESSION_START and SESSION_END:
        day = df["datetime"].dt.date
        s = pd.to_datetime(day.astype(str) + " " + SESSION_START)
        e = pd.to_datetime(day.astype(str) + " " + SESSION_END)
        mask = (df["datetime"] >= s) & (df["datetime"] <= e)
        df = df[mask]

    # Build 1-minute bars
    df_min = df.set_index("datetime").resample("1min").agg({
        "LTP": ["first", "max", "min", "last"],
        "LTQ": "sum",
        "OpenInterest": "last"
    })
    df_min.columns = ["o", "h", "l", "c", "v", "oi"]
    df_min = df_min.dropna(subset=["o", "h", "l", "c"], how="any")

    # Compute the trading date label (first bar date)
    if len(df_min) == 0:
        raise ValueError(f"No bars after resample for {file}")
    ymd = df_min.index[0].strftime("%Y-%m-%d")
    return df_min, symbol, series, ymd

def main():
    files = find_tick_files()
    if not files:
        print("No input CSV files found. Check INPUT_ROOTS.")
        return

    # Where to write outputs
    out_1m_root = OUTPUT_ROOT / "ohlcv_1m"
    meta_root   = OUTPUT_ROOT / "meta"
    out_1m_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    # We'll build a daily volume table for front-month selection
    daily_rows = []  # dicts: {date, symbol, series, vol}

    for f in files:
        try:
            df_1m, symbol, series, ymd = load_and_aggregate_1m(f)
        except Exception as e:
            print(f"[WARN] Skipped {f}: {e}")
            continue

        # Write per-day per-series 1m bars
        out_dir = out_1m_root / f"symbol={symbol}" / f"series={series}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"date={ymd}.csv"
        df_1m.to_csv(out_path, index=True, date_format="%Y-%m-%d %H:%M:%S")

        # Daily volume for mapping (sum v)
        daily_rows.append({
            "date": ymd, "symbol": symbol, "series": series, "total_v": float(df_1m["v"].sum())
        })

        print(f"[OK] 1m bars -> {out_path}")

    # Build the front-month (by daily volume) selection map
    if daily_rows:
        dfv = pd.DataFrame(daily_rows)
        # pick series with max total_v per (date, symbol)
        idx = dfv.groupby(["date", "symbol"])["total_v"].idxmax()
        front = dfv.loc[idx, ["date", "symbol", "series", "total_v"]].sort_values(["date", "symbol"])
        front.to_csv(meta_root / "front_month_map.csv", index=False)
        print(f"[OK] Front-month map -> {meta_root / 'front_month_map.csv'}")
    else:
        print("[WARN] No daily volume rows produced; check inputs.")

if __name__ == "__main__":
    main()
