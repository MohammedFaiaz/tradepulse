# fill_with_upstox_requests.py
from __future__ import annotations
import json, time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests

DATA_CONT = Path(r"C:\tradePulse\data\continuous")
SECRETS = Path(r"C:\tradePulse\.secrets\token.json")

SESSION_START = "09:15:00"
SESSION_END   = "15:30:00"
SYMBOLS = ("BANKNIFTY","NIFTY")

# Upstox public BOD instruments JSON (docs link “Instruments” page)
INSTRUMENTS_JSON_URL = "https://assets.upstox.com/instruments/instruments.json"  # contains NSE_FO futures with instrument_key

def load_access_token() -> str:
    obj = json.loads(SECRETS.read_text(encoding="utf-8"))
    tok = obj.get("access_token") or obj.get("accessToken") or obj.get("token")
    if not tok:
        raise RuntimeError("No access_token in .secrets\\token.json")
    return tok

def month_range(start_ymd: str, end_ymd: Optional[str]=None) -> List[Tuple[int,int]]:
    start = datetime.fromisoformat(start_ymd).date()
    end = date.today() if end_ymd is None else datetime.fromisoformat(end_ymd).date()
    out=[]; y,m=start.year,start.month
    while (y<end.year) or (y==end.year and m<=end.month):
        out.append((y,m))
        y,m = (y+1,1) if m==12 else (y,m+1)
    return out

def month_start_end(y:int,m:int)->Tuple[date,date]:
    first=date(y,m,1)
    last = (date(y+1,1,1)-timedelta(days=1)) if m==12 else (date(y,m+1,1)-timedelta(days=1))
    return first,last

def ist_session_filter(df: pd.DataFrame) -> pd.DataFrame:
    day = df.index.date.astype("datetime64[D]")
    s = pd.to_datetime(day.astype(str)+" "+SESSION_START)
    e = pd.to_datetime(day.astype(str)+" "+SESSION_END)
    return df[(df.index>=s)&(df.index<=e)]

# ---------- instrument universe via BOD JSON ----------
@dataclass
class FutInfo:
    instrument_key: str
    tradingsymbol: str
    expiry: date

def load_fut_universe() -> Dict[str,List[FutInfo]]:
    r = requests.get(INSTRUMENTS_JSON_URL, timeout=60)
    r.raise_for_status()
    doc = r.json()

    # BOD JSON groups by segment types; futures likely under "FUTURES" or similar
    # We’ll scan all objects and filter by fields that exist in current JSON format:
    out = {"BANKNIFTY": [], "NIFTY": []}

    def parse_entry(e: dict):
        ts = str(e.get("tradingsymbol") or e.get("trading_symbol") or "")
        ik = str(e.get("instrument_key") or e.get("instrumentKey") or "")
        exp = e.get("expiry") or e.get("expiry_date")
        typ = (e.get("instrument_type") or e.get("instrumentType") or "")
        ul  = str(e.get("underlying_symbol") or e.get("underlyingSymbol") or "")
        if not (ts and ik and exp):
            return None
        try:
            expd = datetime.fromisoformat(str(exp)).date()
        except Exception:
            return None
        base = "BANKNIFTY" if ("BANKNIFTY" in ts or ul=="BANKNIFTY") else ("NIFTY" if ("NIFTY" in ts or ul=="NIFTY") else None)
        if not base: return None
        if ("FUT" in ts) or (isinstance(typ,str) and typ.upper().startswith("FUT")):
            return base, FutInfo(ik, ts, expd)
        return None

    # The JSON may have multiple lists; flatten
    def walk(x):
        if isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for e in x:
                if isinstance(e, (dict,list)): 
                    res = parse_entry(e) if isinstance(e, dict) else None
                    if res:
                        base, fi = res
                        out[base].append(fi)
                walk(e) if isinstance(e, (list,dict)) else None

    walk(doc)
    for k in out: out[k].sort(key=lambda f: f.expiry)
    return out

def resolve_front_month(universe: List[FutInfo], day: date) -> FutInfo:
    cands = [fi for fi in universe if fi.expiry >= day]
    return min(cands, key=lambda x: x.expiry) if cands else max(universe, key=lambda x: x.expiry)

# ---------- historical candles V3 (minutes/{n}) ----------
def fetch_month_candles_v3(token: str, instrument_key: str, y:int, m:int, minutes:int) -> pd.DataFrame:
    first,last = month_start_end(y,m)
    url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/{minutes}/{last.isoformat()}/{first.isoformat()}"
    headers = {"Accept":"application/json","Authorization":f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code==404:
        return pd.DataFrame(columns=["o","h","l","c","v","oi","data_source"])
    resp.raise_for_status()
    data = resp.json()
    candles = data.get("data", {}).get("candles", [])
    rows=[]
    for c in candles:
        # format: [timestamp, open, high, low, close, volume, open_interest]
        t = pd.to_datetime(c[0])
        o,h,l,cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
        v = float(c[5]) if len(c)>5 and c[5] is not None else None
        oi= float(c[6]) if len(c)>6 and c[6] is not None else None
        rows.append((t,o,h,l,cl,v,oi))
    df = pd.DataFrame(rows, columns=["t","o","h","l","c","v","oi"]).set_index("t").sort_index()
    df = ist_session_filter(df)
    df["data_source"]="upstox_hist"
    return df

def append_dedupe_csv(dest: Path, new_rows: pd.DataFrame):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        old = pd.read_csv(dest, parse_dates=["t"]).set_index("t")
        combo = pd.concat([old, new_rows]).sort_index()
        combo = combo[~combo.index.duplicated(keep="last")]
    else:
        combo = new_rows
    combo.to_csv(dest, index=True, date_format="%Y-%m-%d %H:%M:%S")

@dataclass
class Plan:
    start_ymd: str
    end_ymd: Optional[str] = None
    minutes_list: Tuple[int, ...] = (1,3,5)   # 1m always; include 3,5 if you want direct fetch
    sleep_sec: float = 0.7

def run_gap_fill(plan: Plan):
    token = load_access_token()
    uni = load_fut_universe()
    months = month_range(plan.start_ymd, plan.end_ymd)
    if not months:
        print("No months to fetch."); return

    for symbol in SYMBOLS:
        print(f"\n=== {symbol}: {len(months)} month(s) ===")
        for (y,m) in months:
            mid = month_start_end(y,m)[0] + timedelta(days=10)
            try:
                front = resolve_front_month(uni[symbol], mid)
            except Exception as e:
                print(f"[WARN] {symbol} {y}-{m:02d}: cannot resolve front month: {e}")
                continue
            print(f"Month {y}-{m:02d} -> {front.tradingsymbol} ({front.expiry})")

            for mins in plan.minutes_list:
                df = fetch_month_candles_v3(token, front.instrument_key, y, m, minutes=mins)
                if df.empty:
                    print(f"[MISS] {symbol} {y}-{m:02d} {mins}m: no data")
                    continue
                dest = DATA_CONT / f"{symbol}_CONT_{mins}m.csv"
                append_dedupe_csv(dest, df)
                print(f"[OK]  appended {len(df)} rows -> {dest.name}")
                time.sleep(plan.sleep_sec)

    print("\nDone.")
    print("If you fetched only 1m, you can resample to 3m/5m with your existing script.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fill continuous files using Upstox REST (no SDK).")
    ap.add_argument("--start", default="2025-05-28")
    ap.add_argument("--end", default=None)
    ap.add_argument("--minutes", nargs="+", type=int, default=[1,3,5], choices=[1,3,5])
    ap.add_argument("--sleep", type=float, default=0.7)
    args = ap.parse_args()
    run_gap_fill(Plan(start_ymd=args.start, end_ymd=args.end, minutes_list=tuple(args.minutes), sleep_sec=args.sleep))
