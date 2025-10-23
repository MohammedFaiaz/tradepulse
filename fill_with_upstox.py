# fill_with_upstox.py
from __future__ import annotations
import json
import time
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Upstox v3 SDK
from upstox_client import ApiClient, Configuration
from upstox_client.apis.instruments_api import InstrumentsApi
from upstox_client.apis.market_quote_api import MarketQuoteApi

DATA_CONT = Path(r"C:\tradePulse\data\continuous")
SECRETS = Path(r"C:\tradePulse\.secrets\token.json")

SESSION_START = "09:15:00"
SESSION_END   = "15:30:00"

SYMBOLS = ("BANKNIFTY", "NIFTY")
# Map our TF strings -> Upstox "interval" values
UPSTOX_INTERVAL = {
    "1m":  "1minute",
    "3m":  "3minute",
    "5m":  "5minute",
}

# ---------- helpers ----------

def load_access_token() -> str:
    if not SECRETS.exists():
        raise FileNotFoundError(f"Missing token file: {SECRETS}")
    obj = json.loads(SECRETS.read_text(encoding="utf-8"))
    token = obj.get("access_token") or obj.get("accessToken") or obj.get("token")
    if not token:
        raise ValueError("access_token not found in token.json")
    return token

def month_range(start_ymd: str, end_ymd: Optional[str] = None) -> List[Tuple[int,int]]:
    """Inclusive list of (year, month) from start to end (today if None)."""
    start = datetime.fromisoformat(start_ymd).date()
    end = date.today() if end_ymd is None else datetime.fromisoformat(end_ymd).date()
    out = []
    y, m = start.year, start.month
    while (y < end.year) or (y == end.year and m <= end.month):
        out.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out

def month_start_end(y: int, m: int) -> Tuple[date, date]:
    """Return (first_day, last_day) of a month as dates."""
    first = date(y, m, 1)
    if m == 12:
        last = date(y+1, 1, 1) - timedelta(days=1)
    else:
        last = date(y, m+1, 1) - timedelta(days=1)
    return first, last

def ist_session_filter(df: pd.DataFrame) -> pd.DataFrame:
    # assumes timestamps are IST naive or already converted
    day = df.index.date.astype("datetime64[D]")
    s = pd.to_datetime(day.astype(str) + " " + SESSION_START)
    e = pd.to_datetime(day.astype(str) + " " + SESSION_END)
    mask = (df.index >= s) & (df.index <= e)
    return df.loc[mask]

def append_dedupe_csv(dest: Path, new_rows: pd.DataFrame):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        old = pd.read_csv(dest, parse_dates=["t"]).set_index("t")
        combo = pd.concat([old, new_rows]).sort_index()
        combo = combo[~combo.index.duplicated(keep="last")]
    else:
        combo = new_rows
    combo.to_csv(dest, index=True, date_format="%Y-%m-%d %H:%M:%S")

# ---------- instrument resolution ----------

def build_clients(token: str) -> Tuple[InstrumentsApi, MarketQuoteApi]:
    cfg = Configuration(access_token=token)
    api_client = ApiClient(cfg)
    return InstrumentsApi(api_client), MarketQuoteApi(api_client)

@dataclass
class FutInfo:
    instrument_key: str
    tradingsymbol: str
    expiry: date

def load_fut_universe(inst_api: InstrumentsApi) -> Dict[str, List[FutInfo]]:
    """
    Pull NSE_FO instruments and keep only index futures for BANKNIFTY/NIFTY.
    Returns dict: { "BANKNIFTY": [FutInfo...], "NIFTY": [FutInfo...] }
    """
    universe = {"BANKNIFTY": [], "NIFTY": []}
    # get_instruments(exchange) returns a list of dicts; we filter FUTIDX for our underlyings
    instruments = inst_api.get_instruments("NSE_FO")
    for ins in instruments:
        # defensively read keys that usually exist
        ts = str(ins.get("tradingsymbol", ""))
        key = str(ins.get("instrument_key", ""))
        expiry_str = ins.get("expiry")
        inst_type = ins.get("instrument_type", "")  # often "FUTIDX"
        ul = str(ins.get("underlying_symbol", ""))  # often "NIFTY" or "BANKNIFTY"

        if not (key and ts and expiry_str):
            continue
        try:
            exp = datetime.fromisoformat(expiry_str).date()
        except Exception:
            continue

        # keep only index futures on our two symbols
        # Different accounts may see slightly different strings; be generous:
        if ul in ("NIFTY", "BANKNIFTY") or ts.startswith("NIFTY") or ts.startswith("BANKNIFTY"):
            if "FUT" in ts or inst_type.startswith("FUT"):
                base = "BANKNIFTY" if "BANKNIFTY" in ts or ul == "BANKNIFTY" else ("NIFTY" if "NIFTY" in ts or ul == "NIFTY" else None)
                if base:
                    universe[base].append(FutInfo(key, ts, exp))

    # sort ascending by expiry
    for k in universe:
        universe[k].sort(key=lambda x: x.expiry)
    return universe

def resolve_front_month(universe: List[FutInfo], day: date) -> FutInfo:
    """
    Choose the front-month contract for a given day:
    - If there exists a future with expiry >= day, choose the nearest such expiry.
    - Else choose the one with the latest expiry prior to day (expired historical).
    """
    candidates = [fi for fi in universe if fi.expiry >= day]
    if candidates:
        return min(candidates, key=lambda x: x.expiry)
    # fallback (rare for very old dates)
    return max(universe, key=lambda x: x.expiry)

# ---------- historical fetch ----------

def fetch_month_candles(mkt_api: MarketQuoteApi,
                        instrument_key: str,
                        y: int, m: int,
                        tf: str = "1m") -> pd.DataFrame:
    """
    Fetch one month of candles for an instrument_key.
    Returns df indexed by 't' with columns: o,h,l,c,v,oi,data_source
    """
    first, last = month_start_end(y, m)
    interval = UPSTOX_INTERVAL[tf]  # "1minute" / "3minute"/ "5minute"

    # Upstox expects ISO strings; keep inclusive within the month
    resp = mkt_api.get_historical_candle_data(
        instrument_key=instrument_key,
        interval=interval,
        from_date=first.isoformat(),
        to_date=last.isoformat()
    )

    # The SDK returns resp.data.candles (usually list of lists)
    candles = getattr(resp, "data", {}).get("candles", []) if hasattr(resp, "data") else []
    if not candles:
        return pd.DataFrame(columns=["o","h","l","c","v","oi","data_source"])

    # Expected order per candle: [timestamp, open, high, low, close, volume, oi]
    rows = []
    for c in candles:
        # be defensive about len
        ts = c[0]
        o = float(c[1]); h = float(c[2]); l = float(c[3]); cl = float(c[4])
        v = None
        oi = None
        if len(c) > 5:
            try: v = float(c[5])
            except: v = None
        if len(c) > 6:
            try: oi = float(c[6])
            except: oi = None

        # parse timestamp as naive (assume exchange local time for consistency with your files)
        t = pd.to_datetime(ts)
        rows.append((t, o, h, l, cl, v, oi))

    df = pd.DataFrame(rows, columns=["t","o","h","l","c","v","oi"]).set_index("t").sort_index()

    # Session filter 09:15–15:30
    df = ist_session_filter(df)

    # Mark provenance
    df["data_source"] = "upstox_hist"
    return df

# ---------- orchestrator ----------

@dataclass
class Plan:
    start_ymd: str
    end_ymd: Optional[str] = None
    timeframes: Tuple[str, ...] = ("1m",)       # always include 1m; optionally add "3m","5m"
    sleep_sec: float = 0.7                      # gentle pacing between calls

def run_gap_fill(plan: Plan):
    token = load_access_token()
    inst_api, mkt_api = build_clients(token)
    uni = load_fut_universe(inst_api)

    months = month_range(plan.start_ymd, plan.end_ymd)
    if not months:
        print("Nothing to fetch; check dates.")
        return

    for symbol in SYMBOLS:
        print(f"\n=== {symbol}: planning {len(months)} month(s) ===")
        cont_paths = {tf: (DATA_CONT / f"{symbol}_CONT_{tf}.csv") for tf in plan.timeframes}

        for (y, m) in months:
            first, last = month_start_end(y, m)
            # choose a mid-month day to resolve the contract, then re-use for whole month
            mid_day = first + timedelta(days=10)
            try:
                front = resolve_front_month(uni[symbol], mid_day)
            except Exception as e:
                print(f"[WARN] Could not resolve front-month for {symbol} {y}-{m:02d}: {e}")
                continue

            print(f"Fetching {symbol} {y}-{m:02d}  -> {front.tradingsymbol}  ({front.expiry})")

            for tf in plan.timeframes:
                try:
                    df = fetch_month_candles(mkt_api, front.instrument_key, y, m, tf=tf)
                except Exception as e:
                    print(f"[ERR] {symbol} {y}-{m:02d} {tf}: {e}")
                    continue

                if df.empty:
                    print(f"[MISS] {symbol} {y}-{m:02d} {tf}: no candles")
                    continue

                # Append + de-dup
                append_dedupe_csv(cont_paths[tf], df)
                print(f"[OK]  appended {len(df)} rows -> {cont_paths[tf].name}")

                time.sleep(plan.sleep_sec)

    print("\nDone. You can now (re)run your backtests on the extended range.")
    print("If you fetched only 1m, remember you can resample 1m → 3m/5m with your existing script.")

# ---------- CLI ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fill continuous files using Upstox historical candles (front-month futures).")
    ap.add_argument("--start", required=False, default="2025-05-28", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end",   required=False, default=None,         help="YYYY-MM-DD (inclusive). Default = today.")
    ap.add_argument("--tfs",   nargs="+",      default=["1m","3m","5m"], choices=["1m","3m","5m"], help="Timeframes to fetch directly.")
    ap.add_argument("--sleep", type=float,     default=0.7, help="Sleep seconds between API calls.")
    args = ap.parse_args()

    run_gap_fill(Plan(start_ymd=args.start, end_ymd=args.end, timeframes=tuple(args.tfs), sleep_sec=args.sleep))
