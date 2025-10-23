# build_continuous_and_resample.py
# Uses per-series 1m bars + front_month_map.csv to build continuous 1m for each symbol,
# then resamples to 3m and 5m. Appends safely (idempotent by datetime).

from __future__ import annotations
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ==== CONFIG ====
DATA_ROOT = Path(r"C:\tradePulse\data")
SYMBOLS = ("BANKNIFTY", "NIFTY")
# ==============

def load_front_map() -> pd.DataFrame:
    f = DATA_ROOT / "meta" / "front_month_map.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing front_month_map.csv at {f}. Run ingest_csv_to_1m.py first.")
    df = pd.read_csv(f, dtype={"date": "string", "symbol": "string", "series": "string"})
    return df

def read_daily_1m(symbol: str, series: str, date: str) -> pd.DataFrame:
    p = DATA_ROOT / "ohlcv_1m" / f"symbol={symbol}" / f"series={series}" / f"date={date}.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    # Read without parse_dates first, then detect the time column name
    df = pd.read_csv(p)
    # Try common header names in order of preference
    time_col = None
    for cand in ("t", "datetime", "Timestamp", "time", "Time", "Unnamed: 0"):
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        # fall back to "first column is time"
        time_col = df.columns[0]

    # Normalize to a proper datetime index named "t"
    df["t"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["t"]).set_index("t")

    # Ensure expected columns exist
    for col in ("o","h","l","c","v"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {p}")

    # OI might be absent in some files; if so, create it as NaN
    if "oi" not in df.columns:
        df["oi"] = pd.NA

    # Tag series for later reference
    df["series"] = series

    # Keep canonical column order
    return df[["o","h","l","c","v","oi","series"]]


def append_unique(dest: Path, df: pd.DataFrame):
    """Append rows by datetime, dropping duplicates if file exists."""
    if dest.exists():
        old = pd.read_csv(dest, parse_dates=["t"]).set_index("t")
        combo = pd.concat([old, df]).sort_index()
        combo = combo[~combo.index.duplicated(keep="last")]
    else:
        combo = df
    combo.to_csv(dest, index=True, date_format="%Y-%m-%d %H:%M:%S")

def resample_and_write(src_csv: Path, rule: str, out_csv: Path):
    if not src_csv.exists():
        return
    df = pd.read_csv(src_csv, parse_dates=["t"]).set_index("t")
    agg = {
        "o":"first","h":"max","l":"min","c":"last","v":"sum","oi":"last"
    }
    # keep the most recent series label in each resampled bar
    def last_series(x):
        return x.dropna().iloc[-1] if len(x.dropna()) else None

    out = df.resample(rule).agg({**agg, "series": last_series})
    out = out.dropna(subset=["o","h","l","c"], how="any")
    out.to_csv(out_csv, index=True, date_format="%Y-%m-%d %H:%M:%S")

def main():
    fm = load_front_map()
    cont_root = DATA_ROOT / "continuous"
    cont_root.mkdir(parents=True, exist_ok=True)

    for symbol in SYMBOLS:
        cont_1m = cont_root / f"{symbol}_CONT_1m.csv"
        # Build/append 1m continuous by iterating dates in map
        sub = fm[fm["symbol"] == symbol].sort_values("date")
        if sub.empty:
            print(f"[WARN] No rows in front_month_map for {symbol}")
            continue

        for _, row in sub.iterrows():
            date = row["date"]
            series = row["series"]
            try:
                df_day = read_daily_1m(symbol, series, date)
            except FileNotFoundError:
                print(f"[MISS] {symbol} {series} {date} missing 1m file; skip.")
                continue
            append_unique(cont_1m, df_day)
            print(f"[OK] appended {symbol} {series} {date} -> {cont_1m.name}")

        # Resample to 3m and 5m from the continuous 1m
        resample_and_write(cont_1m, "3min", cont_root / f"{symbol}_CONT_3m.csv")
        resample_and_write(cont_1m, "5min", cont_root / f"{symbol}_CONT_5m.csv")
        print(f"[OK] resampled -> {symbol}_CONT_3m.csv / {symbol}_CONT_5m.csv")

if __name__ == "__main__":
    main()
