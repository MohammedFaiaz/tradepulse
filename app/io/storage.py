# app/io/storage.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

DATA_ROOT = Path(r"C:\tradePulse\data\continuous")  # adjust if needed

SESSION_START = "09:15:00"
SESSION_END   = "15:30:00"

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    # Accept either 't' or unnamed first column as timestamp
    if "t" in df.columns:
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna(subset=["t"]).set_index("t")
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        df = df.dropna(subset=[first_col]).set_index(first_col)
        df.index.name = "t"
    return df

def _session_filter(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only session bars (IST 09:15–15:30); assumes timestamps are local/IST
    day = df.index.date.astype("datetime64[D]")
    s = pd.to_datetime(day.astype(str) + " " + SESSION_START)
    e = pd.to_datetime(day.astype(str) + " " + SESSION_END)
    mask = (df.index >= s) & (df.index <= e)
    return df.loc[mask]

def load_cont(symbol: str, tf: str = "1m") -> pd.DataFrame:
    """
    Load continuous CSV created earlier, with columns: o,h,l,c,v,oi[,series][,data_source].
    Returns a DataFrame indexed by 't' and session-filtered.
    """
    p = DATA_ROOT / f"{symbol}_CONT_{tf}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df = _ensure_datetime_index(df)
    # Ensure expected columns exist
    for col in ["o","h","l","c"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {p}")
    # Optional columns
    if "v" not in df.columns: df["v"] = pd.NA
    if "oi" not in df.columns: df["oi"] = pd.NA
    if "data_source" not in df.columns: df["data_source"] = pd.NA
    # Session filter
    df = _session_filter(df)
    # Sort & de-dup
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def save_parquet(df: pd.DataFrame, path: str | Path):
    """Utility if you want to convert CSV→Parquet for speed later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
