# fill_with_upstox_requests_v2.py
from __future__ import annotations
import json
import time
import gzip
from io import BytesIO
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests

# ---------- Paths & constants ----------
DATA_CONT = Path(r"C:\tradePulse\data\continuous")
SECRETS   = Path(r"C:\tradePulse\.secrets\token.json")
META_DIR  = Path(r"C:\tradePulse\data\meta")
META_DIR.mkdir(parents=True, exist_ok=True)

SESSION_START = "09:15:00"
SESSION_END   = "15:30:00"
SYMBOLS = ("BANKNIFTY", "NIFTY")

API_BASE = "https://api.upstox.com"

# Public BOD (instrument master) gz JSONs, per Upstox docs
BOD_URLS = [
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",       # NSE only
    "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz",  # All exchanges
]

# ---------- Small utilities ----------
def load_access_token() -> str:
    if not SECRETS.exists():
        raise FileNotFoundError(f"Missing token file: {SECRETS}")
    obj = json.loads(SECRETS.read_text(encoding="utf-8"))
    tok = obj.get("access_token") or obj.get("accessToken") or obj.get("token")
    if not tok:
        raise RuntimeError("No access_token in .secrets\\token.json")
    return tok

def month_range(start_ymd: str, end_ymd: Optional[str] = None) -> List[Tuple[int, int]]:
    start = datetime.fromisoformat(start_ymd).date()
    end = date.today() if end_ymd is None else datetime.fromisoformat(end_ymd).date()
    out = []
    y, m = start.year, start.month
    while (y < end.year) or (y == end.year and m <= end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out

def month_start_end(y: int, m: int) -> Tuple[date, date]:
    first = date(y, m, 1)
    last = (date(y + 1, 1, 1) - timedelta(days=1)) if m == 12 else (date(y, m + 1, 1) - timedelta(days=1))
    return first, last

def ist_session_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to naive IST and keep only 09:15–15:30."""
    if df.empty:
        return df
    # Normalize tz-aware index to naive IST
    if getattr(df.index, "tz", None) is not None:
        try:
            df = df.tz_convert("Asia/Kolkata")
        except Exception:
            # Some payloads may be tz-naive but with offset: coerce first
            df = df.tz_localize("Asia/Kolkata", nonexistent="shift_forward", ambiguous="NaT")
        df.index = df.index.tz_localize(None)

    day = df.index.date.astype("datetime64[D]")
    s = pd.to_datetime(day.astype(str) + " " + SESSION_START)
    e = pd.to_datetime(day.astype(str) + " " + SESSION_END)
    mask = (df.index >= s) & (df.index <= e)
    return df.loc[mask]

def append_dedupe_csv(dest: Path, new_rows: pd.DataFrame):
    """Append to CSV and de-duplicate by timestamp."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        old = pd.read_csv(dest, parse_dates=["t"]).set_index("t")
        combo = pd.concat([old, new_rows]).sort_index()
        combo = combo[~combo.index.duplicated(keep="last")]
    else:
        combo = new_rows
    combo.to_csv(dest, index=True, date_format="%Y-%m-%d %H:%M:%S")

# ---------- Auth’d request helper ----------
def _req_json(url: str, token: str, params: Optional[dict] = None) -> dict:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, params=params or {}, timeout=60)
    if r.status_code == 200:
        return r.json()
    raise requests.HTTPError(f"{r.status_code} for {url}: {r.text}")

# ---------- Instrument universe (BOD gz → fallbacks) ----------
@dataclass
class FutInfo:
    instrument_key: str
    tradingsymbol: str
    expiry: date
    underlying: str

def _load_instruments_any(token: str) -> List[dict]:
    """
    Try public BOD gz JSON first.
    Then authenticated v2 endpoints.
    Then local file fallback.
    """
    # 1) Public BOD (gz JSON)
    for url in BOD_URLS:
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.content:
                buf = BytesIO(r.content)
                with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
                    raw = json.loads(gz.read().decode("utf-8"))
                # Normalize to list
                if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
                    return raw["data"]
                if isinstance(raw, list):
                    return raw
        except Exception:
            pass  # try next

    # 2) Authenticated v2 fallbacks (some accounts don’t have these)
    tried = []
    for url, params in [
        (f"{API_BASE}/v2/instruments", {"segment": "NSE_FO"}),
        (f"{API_BASE}/v2/instruments/derivatives", {"exchange": "NSE_FO"}),
    ]:
        try:
            tried.append(url)
            data = _req_json(url, token, params)
            payload = data.get("data", data)
            if isinstance(payload, list) and payload:
                return payload
        except Exception:
            continue

    # 3) Local dump fallback
    local = META_DIR / "instruments.json"
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))

    raise RuntimeError(
        "Could not load instruments from public gz, v2 API, or local file.\n"
        f"Tried: {BOD_URLS + tried}\n"
        f"If needed, download BOD JSON and save as {local}"
    )

def load_fut_universe(token: str) -> Dict[str, List[FutInfo]]:
    raw = _load_instruments_any(token)
    out: Dict[str, List[FutInfo]] = {"BANKNIFTY": [], "NIFTY": []}

    def to_date(exp_val):
        """Support ISO strings or epoch milliseconds."""
        if isinstance(exp_val, (int, float)):
            try:
                return datetime.utcfromtimestamp(float(exp_val) / 1000.0).date()
            except Exception:
                return None
        try:
            return datetime.fromisoformat(str(exp_val)).date()
        except Exception:
            return None

    for e in raw:
        ts  = str(e.get("tradingsymbol") or e.get("trading_symbol") or e.get("symbol") or "")
        ik  = str(e.get("instrument_key") or e.get("instrumentKey") or e.get("token") or "")
        exp = e.get("expiry") or e.get("expiry_date")
        typ = (e.get("instrument_type") or e.get("instrumentType") or "")
        ul  = str(e.get("underlying_symbol") or e.get("underlyingSymbol") or e.get("underlying") or "")
        seg = str(e.get("segment") or "")

        if not (ts and ik and exp):
            continue
        # Keep NSE_FO where present; if seg absent we keep and filter by symbol/type
        if seg and seg != "NSE_FO":
            continue

        expd = to_date(exp)
        if not expd:
            continue

        base = None
        if "BANKNIFTY" in ts or ul == "BANKNIFTY":
            base = "BANKNIFTY"
        elif "NIFTY" in ts or ul == "NIFTY":
            base = "NIFTY"
        if not base:
            continue

        # FUT filter
        if "FUT" in ts or (isinstance(typ, str) and typ.upper().startswith("FUT")):
            out[base].append(FutInfo(ik, ts, expd, base))

    for k in out:
        out[k].sort(key=lambda f: f.expiry)

    # Save an audit CSV for your reference
    (META_DIR / "upstox_contracts_map.csv").write_text(
        "symbol,tradingsymbol,instrument_key,expiry\n" +
        "\n".join(f"{k},{fi.tradingsymbol},{fi.instrument_key},{fi.expiry.isoformat()}"
                  for k in out for fi in out[k]),
        encoding="utf-8"
    )

    if not out["BANKNIFTY"] and not out["NIFTY"]:
        raise RuntimeError("No BANKNIFTY/NIFTY futures found in instruments payload (NSE_FO FUT).")

    return out

def resolve_contract_for_month(universe: List[FutInfo], y: int, m: int) -> FutInfo:
    """Pick the monthly contract whose expiry lies within (y,m); else nearest within +/-45 days."""
    first, last = month_start_end(y, m)
    in_month = [fi for fi in universe if first <= fi.expiry <= last]
    if in_month:
        # Choose the nearest-to-expiry within the month
        return min(in_month, key=lambda x: x.expiry)

    # No expiry exactly inside the month: choose nearest expiry within +/- 45 days window
    window_days = 45
    def days_from_month(fi):
        # distance from first day of the month
        return abs((fi.expiry - first).days)

    near = [fi for fi in universe if days_from_month(fi) <= window_days]
    if near:
        return min(near, key=lambda x: days_from_month(x))

    # Fallback: nearest by absolute distance
    return min(universe, key=lambda x: days_from_month(x))

# ---------- Historical candles (V3 REST) ----------
def fetch_month_candles_v3(token: str, instrument_key: str, y: int, m: int, minutes: int) -> pd.DataFrame:
    first, last = month_start_end(y, m)
    url = f"{API_BASE}/v3/historical-candle/{instrument_key}/minutes/{minutes}/{last.isoformat()}/{first.isoformat()}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=60)

    if resp.status_code == 404:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v", "oi", "data_source"])
    resp.raise_for_status()

    data = resp.json()
    candles = data.get("data", {}).get("candles", [])
    rows = []
    for c in candles:
        # [timestamp, open, high, low, close, volume, open_interest]
        t = pd.to_datetime(c[0])
        o, h, l, cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
        v  = float(c[5]) if len(c) > 5 and c[5] is not None else None
        oi = float(c[6]) if len(c) > 6 and c[6] is not None else None
        rows.append((t, o, h, l, cl, v, oi))

    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v", "oi"]).set_index("t").sort_index()

    # --- Normalize tz (if any) to naive IST, then apply session filter ---
    if getattr(df.index, "tz", None) is not None:
        try:
            df = df.tz_convert("Asia/Kolkata")
        except Exception:
            df = df.tz_localize("Asia/Kolkata", nonexistent="shift_forward", ambiguous="NaT")
        df.index = df.index.tz_localize(None)

    df = ist_session_filter(df)
    df["data_source"] = "upstox_hist"
    return df

# ---------- Orchestrator ----------
@dataclass
class Plan:
    start_ymd: str
    end_ymd: Optional[str] = None
    minutes_list: Tuple[int, ...] = (1, 3, 5)
    sleep_sec: float = 0.7

def run_gap_fill(plan: Plan):
    token = load_access_token()
    uni = load_fut_universe(token)
    months = month_range(plan.start_ymd, plan.end_ymd)
    if not months:
        print("No months to fetch.")
        return

    for symbol in SYMBOLS:
        print(f"\n=== {symbol}: {len(months)} month(s) ===")
        for (y, m) in months:
            try:
                front = resolve_contract_for_month(uni[symbol], y, m)
            except Exception as e:
                print(f"[WARN] {symbol} {y}-{m:02d}: cannot resolve monthly contract: {e}")
                continue

            print(f"Month {y}-{m:02d} -> {front.tradingsymbol} ({front.expiry})")

            for mins in plan.minutes_list:
                try:
                    df = fetch_month_candles_v3(token, front.instrument_key, y, m, minutes=mins)
                except Exception as e:
                    print(f"[ERR]  {symbol} {y}-{m:02d} {mins}m: {e}")
                    continue

                if df.empty:
                    print(f"[MISS] {symbol} {y}-{m:02d} {mins}m: no data")
                    continue

                dest = DATA_CONT / f"{symbol}_CONT_{mins}m.csv"
                append_dedupe_csv(dest, df)
                print(f"[OK]   appended {len(df)} rows -> {dest.name}")
                time.sleep(plan.sleep_sec)

    print("\nDone. Re-run backtests on the extended range when ready.")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fill continuous files via authenticated Upstox REST.")
    ap.add_argument("--start", default="2025-05-28", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (inclusive). Default: today")
    ap.add_argument("--minutes", nargs="+", type=int, default=[1, 3, 5], choices=[1, 3, 5], help="Timeframes to fetch directly")
    ap.add_argument("--sleep", type=float, default=0.7, help="Sleep seconds between API calls")
    args = ap.parse_args()

    run_gap_fill(Plan(start_ymd=args.start, end_ymd=args.end, minutes_list=tuple(args.minutes), sleep_sec=args.sleep))
