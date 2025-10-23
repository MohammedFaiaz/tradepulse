# fill_with_upstox_requests_v3.py
from __future__ import annotations
import json, time, gzip, calendar
from io import BytesIO
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
from urllib.parse import quote as urlquote

# ---------- Paths & constants ----------
DATA_CONT = Path(r"C:\tradePulse\data\continuous")
SECRETS   = Path(r"C:\tradePulse\.secrets\token.json")
META_DIR  = Path(r"C:\tradePulse\data\meta")
MANUAL_KEYS = META_DIR / "manual_keys.json"
META_DIR.mkdir(parents=True, exist_ok=True)

SESSION_START = (9, 15)   # IST open
SESSION_END   = (15, 30)  # IST close
SYMBOLS = ("BANKNIFTY", "NIFTY")
EXCLUDE_UNDERLYINGS = {"FINNIFTY", "MIDCPNIFTY"}
API_BASE = "https://api.upstox.com"

BOD_URLS = [
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz",
]

# ---------- NSE Monthly expiry rules ----------
TRADING_HOLIDAYS: set[str] = set()  # fill if you want to skip real holidays
CUTOFF_NEW_RULE = date(2025, 9, 1)  # expiries on/after this use Tuesday

def _last_weekday_of_month(y: int, m: int, weekday: int) -> date:
    last_day = calendar.monthrange(y, m)[1]
    for d in range(last_day, 0, -1):
        dd = date(y, m, d)
        if dd.weekday() == weekday:
            return dd
    raise RuntimeError("No weekday found?")

def _is_holiday(d: date) -> bool:
    return d.strftime("%Y-%m-%d") in TRADING_HOLIDAYS

def _roll_back_to_prev_trading_day(d: date) -> date:
    while _is_holiday(d) or d.weekday() == 6:  # Sunday
        d -= timedelta(days=1)
        if d.weekday() == 6:
            d -= timedelta(days=1)
    return d

def estimate_monthly_expiry(y: int, m: int) -> date:
    # Before Sep 2025: last Thursday (3). From Sep 2025: last Tuesday (1).
    anchor_weekday = 1 if date(y, m, 1) >= CUTOFF_NEW_RULE else 3
    return _roll_back_to_prev_trading_day(_last_weekday_of_month(y, m, anchor_weekday))

# ---------- Utilities ----------
def load_access_token() -> str:
    obj = json.loads(SECRETS.read_text(encoding="utf-8"))
    tok = obj.get("access_token") or obj.get("accessToken") or obj.get("token")
    if not tok:
        raise RuntimeError("No access_token in .secrets\\token.json")
    return tok

def month_range(start_ymd: str, end_ymd: Optional[str] = None) -> List[Tuple[int, int]]:
    start = datetime.fromisoformat(start_ymd).date()
    end = date.today() if end_ymd is None else datetime.fromisoformat(end_ymd).date()
    out: List[Tuple[int,int]] = []
    y, m = start.year, start.month
    while (y < end.year) or (y == end.year and m <= end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out

def month_start_end(y: int, m: int) -> Tuple[date, date]:
    first = date(y, m, 1)
    last  = (date(y + 1, 1, 1) - timedelta(days=1)) if m == 12 else (date(y, m + 1, 1) - timedelta(days=1))
    return first, last

def month_code(m: int) -> str:
    return ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"][m-1]

def ist_session_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if getattr(df.index, "tz", None) is not None:
        try:
            df = df.tz_convert("Asia/Kolkata")
        except Exception:
            df = df.tz_localize("Asia/Kolkata", nonexistent="shift_forward", ambiguous="NaT")
        df.index = df.index.tz_localize(None)
    start_mask = df.index.normalize() + pd.Timedelta(hours=SESSION_START[0], minutes=SESSION_START[1])
    end_mask   = df.index.normalize() + pd.Timedelta(hours=SESSION_END[0],   minutes=SESSION_END[1])
    return df[(df.index >= start_mask) & (df.index <= end_mask)]

def append_dedupe_csv(dest: Path, new_rows: pd.DataFrame):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        old = pd.read_csv(dest, parse_dates=["t"], low_memory=False).set_index("t")
        combo = pd.concat([old, new_rows]).sort_index()
        combo = combo[~combo.index.duplicated(keep="last")]
    else:
        combo = new_rows
    combo.to_csv(dest, index=True, date_format="%Y-%m-%d %H:%M:%S")

# ---------- HTTP ----------
def _auth_get_json(url: str, token: str, params: Optional[dict] = None) -> dict:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, params=params or {}, timeout=15)
    if r.status_code == 200:
        return r.json()
    raise requests.HTTPError(f"{r.status_code} GET {url} :: {r.text}")

def _load_bod_any() -> List[dict]:
    for url in BOD_URLS:
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.content:
                buf = BytesIO(r.content)
                with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
                    raw = json.loads(gz.read().decode("utf-8"))
                if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
                    return raw["data"]
                if isinstance(raw, list):
                    return raw
        except Exception:
            pass
    return []

# ---------- Models ----------
@dataclass
class FutInfo:
    instrument_key: str   # as provided/normalized
    tradingsymbol: str
    expiry: date
    underlying: str

def _to_date_maybe_iso_or_epoch(exp_val) -> Optional[date]:
    if isinstance(exp_val, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(exp_val)/1000.0).date()
        except Exception:
            return None
    try:
        return datetime.fromisoformat(str(exp_val)).date()
    except Exception:
        return None

def _is_exact_underlying(ts: str, ul_sym: str, base: str) -> bool:
    base = base.upper()
    if base not in {"BANKNIFTY", "NIFTY"}:
        return False
    if any(ex in ts or ex == ul_sym for ex in EXCLUDE_UNDERLYINGS):
        return False
    return (ul_sym == base) or ts.startswith(base)

# ---------- Key normalization & variants ----------
def normalize_instr_key(k: str) -> str:
    if not k:
        return k
    k = k.strip()
    if '|' in k:  # e.g., NSE_FO|NIFTY25OCTFUT
        right = k.split('|', 1)[1].strip()
        return f"NSE:{right}"
    if ':' in k:  # e.g., NSE:NIFTY25OCTFUT
        return k
    return f"NSE:{k}"

def key_variants(raw_key_or_symbol: str) -> List[str]:
    """
    Return an ordered list of instrument_key variants to try against v3:
    1) as given
    2) NSE:<symbol>
    3) NSE_FO|<symbol>
    4) <symbol> (bare)
    """
    s = raw_key_or_symbol.strip()
    parts = [s]
    # Extract symbol only (right side) for recomposition
    symbol_only = s
    if '|' in s:
        symbol_only = s.split('|', 1)[1].strip()
    elif ':' in s:
        symbol_only = s.split(':', 1)[1].strip()

    # Add normalized NSE:SYMBOL
    parts.append(f"NSE:{symbol_only}")
    # Legacy NSE_FO|SYMBOL
    parts.append(f"NSE_FO|{symbol_only}")
    # Bare SYMBOL
    parts.append(symbol_only)
    # Remove dupes, preserve order
    out, seen = [], set()
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

# ---------- Universe loaders ----------
def load_future_universe_from_bod() -> Dict[str, List[FutInfo]]:
    bod = _load_bod_any()
    out = {"BANKNIFTY": [], "NIFTY": []}
    for e in bod:
        ts = str(e.get("tradingsymbol") or e.get("trading_symbol") or e.get("symbol") or "")
        ik_raw = str(e.get("instrument_key") or e.get("instrumentKey") or e.get("token") or "")
        exp = e.get("expiry") or e.get("expiry_date")
        typ = str(e.get("instrument_type") or e.get("instrumentType") or "")
        ul  = str(e.get("underlying_symbol") or e.get("underlyingSymbol") or e.get("underlying") or "")
        seg = str(e.get("segment") or "")

        if not (ts and ik_raw and exp):
            continue
        if seg and seg != "NSE_FO":
            continue
        if not ("FUT" in ts or typ.upper().startswith("FUT")):
            continue
        expd = _to_date_maybe_iso_or_epoch(exp)
        if not expd:
            continue

        for base in SYMBOLS:
            if _is_exact_underlying(ts, ul, base):
                out[base].append(FutInfo(ik_raw, ts, expd, base))
                break

    for k in out:
        out[k].sort(key=lambda f: f.expiry)
    return out

def load_manual_key(symbol: str, y: int, m: int) -> Optional[str]:
    if not MANUAL_KEYS.exists():
        return None
    try:
        data = json.loads(MANUAL_KEYS.read_text(encoding="utf-8"))
        return data.get(symbol, {}).get(f"{y:04d}-{m:02d}")
    except Exception:
        return None

def fetch_instrument_meta(token: str, instrument_key: str) -> Optional[FutInfo]:
    # Try multiple encodings for the /v2/instruments/{key} endpoint too
    for k in key_variants(instrument_key):
        ik_url = urlquote(k, safe="")
        url = f"{API_BASE}/v2/instruments/{ik_url}"
        try:
            data = _auth_get_json(url, token, None)
            payload = data.get("data") or {}
            ts = str(payload.get("tradingsymbol") or payload.get("symbol") or "")
            ul = str(payload.get("underlying_symbol") or payload.get("underlying") or "")
            exp = payload.get("expiry") or payload.get("expiry_date")
            expd = _to_date_maybe_iso_or_epoch(exp)
            if ts and expd:
                base = "BANKNIFTY" if ts.startswith("BANKNIFTY") else ("NIFTY" if ts.startswith("NIFTY") else ul or "")
                return FutInfo(k, ts, expd, base)
        except Exception:
            continue
    return None

def search_month_contract(token: str, base: str, y: int, m: int) -> Optional[FutInfo]:
    yy = y % 100
    mon = month_code(m)
    candidates = [
        f"{base} FUT {mon} {yy}",
        f"{base}{yy}{mon}FUT",
        f"{base} {mon} {yy} FUT",
        f"{base} FUT {yy} {mon}",
        f"{base} {y}{mon} FUT",
        f"{base} {yy}{mon} FUT",
        f"{base} {mon} FUT",
        f"{base}{yy}{mon}",
    ]
    for q in candidates:
        try:
            url = f"{API_BASE}/v2/instruments/search"
            params = {"query": q, "segment": "NSE_FO"}
            data = _auth_get_json(url, token, params)
            payload = data.get("data", [])
            if not isinstance(payload, list) or not payload:
                continue
            first, last = month_start_end(y, m)
            matches: List[FutInfo] = []
            for e in payload:
                ts = str(e.get("tradingsymbol") or e.get("symbol") or "")
                ik_raw = str(e.get("instrument_key") or e.get("token") or "")
                exp = e.get("expiry") or e.get("expiry_date")
                typ = str(e.get("instrument_type") or e.get("instrumentType") or "")
                ul  = str(e.get("underlying_symbol") or e.get("underlying") or "")
                if not (ts and ik_raw and exp):
                    continue
                if not ("FUT" in ts or typ.upper().startswith("FUT")):
                    continue
                if not _is_exact_underlying(ts, ul, base):
                    continue
                expd = _to_date_maybe_iso_or_epoch(exp)
                if not expd:
                    continue
                if first <= expd <= last:
                    matches.append(FutInfo(ik_raw, ts, expd, base))
            if matches:
                return min(matches, key=lambda x: x.expiry)
        except Exception:
            continue
    return None

# ---------- v3 Historical Candles ----------
def _fetch_candles_core(token: str, ik: str, minutes: int, from_date: date, to_date: date) -> requests.Response:
    if to_date < from_date:
        from_date, to_date = to_date, from_date
    ik_url = urlquote(ik, safe="")
    url = f"{API_BASE}/v3/historical-candle/{ik_url}/minutes/{minutes}/{to_date.isoformat()}/{from_date.isoformat()}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    return requests.get(url, headers=headers, timeout=60)

def _parse_candles(resp: requests.Response) -> List[tuple]:
    data = resp.json()
    candles = (data.get("data") or {}).get("candles", [])
    rows = []
    for c in candles:
        # [timestamp, open, high, low, close, volume, open_interest]
        t = pd.to_datetime(c[0])
        o, h, l, cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
        v  = float(c[5]) if len(c) > 5 and c[5] is not None else None
        oi = float(c[6]) if len(c) > 6 and c[6] is not None else None
        rows.append((t, o, h, l, cl, v, oi))
    return rows

def _probe_first_day(token: str, key_try: str, minutes: int, first: date) -> str:
    # Probe a single day to see if API returns data; helpful debug
    resp = _fetch_candles_core(token, key_try, minutes, first, first)
    status = resp.status_code
    rows = 0
    if 200 <= status < 300:
        rows = len(_parse_candles(resp))
    return f"probe[{key_try}] {minutes}m {first} -> status={status}, rows={rows}"

def fetch_month_candles_v3(token: str, fi: FutInfo, y: int, m: int, minutes: int) -> pd.DataFrame:
    first, last = month_start_end(y, m)
    end_clip = min(last, fi.expiry)
    if first > end_clip:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v", "oi", "data_source"])

    # Try with multiple instrument_key variants until we get data
    variants = key_variants(fi.instrument_key)
    # For traceability, print one small probe line (first day) per variant
    for k_try in variants:
        print("   ", _probe_first_day(token, k_try, minutes, first))
        # Try whole month
        resp = _fetch_candles_core(token, k_try, minutes, first, end_clip)
        if resp.status_code == 400:
            resp = _fetch_candles_core(token, k_try, minutes, end_clip, first)
        if resp.status_code == 404:
            continue
        if 200 <= resp.status_code < 300:
            rows = _parse_candles(resp)
            if not rows:
                # fallback to weekly chunks
                chunk_rows: List[tuple] = []
                cur = first
                step = timedelta(days=7)
                while cur <= end_clip:
                    to_ = min(cur + step - timedelta(days=1), end_clip)
                    r = _fetch_candles_core(token, k_try, minutes, cur, to_)
                    if 200 <= r.status_code < 300:
                        chunk_rows.extend(_parse_candles(r))
                    cur = to_ + timedelta(days=1)
                rows = chunk_rows

            if rows:
                df = pd.DataFrame(rows, columns=["t","o","h","l","c","v","oi"]).set_index("t").sort_index()
                # IST normalize & session window
                if getattr(df.index, "tz", None) is not None:
                    try:
                        df = df.tz_convert("Asia/Kolkata")
                    except Exception:
                        df = df.tz_localize("Asia/Kolkata", nonexistent="shift_forward", ambiguous="NaT")
                    df.index = df.index.tz_localize(None)
                df = ist_session_filter(df)
                df["data_source"] = "upstox_hist"
                return df
        # else: try next variant

    # If all variants failed or produced no rows:
    return pd.DataFrame(columns=["o", "h", "l", "c", "v", "oi", "data_source"])

# ---------- Orchestrator ----------
@dataclass
class Plan:
    start_ymd: str
    end_ymd: Optional[str] = None
    minutes_list: Tuple[int, ...] = (1, 3, 5)
    sleep_sec: float = 0.7

def run_gap_fill(plan: Plan):
    token = load_access_token()
    bod_uni = load_future_universe_from_bod()
    months = month_range(plan.start_ymd, plan.end_ymd)
    if not months:
        print("No months to fetch."); return

    for symbol in SYMBOLS:
        print(f"\n=== {symbol}: {len(months)} month(s) ===")
        for (y, m) in months:
            first, last = month_start_end(y, m)
            fi: Optional[FutInfo] = None

            # Manual override first
            manual_key = load_manual_key(symbol, y, m)
            if manual_key:
                meta = fetch_instrument_meta(token, manual_key)
                if meta:
                    fi = meta
                else:
                    fi = FutInfo(manual_key, f"{symbol} MANUAL {y}-{m:02d}", estimate_monthly_expiry(y, m), symbol)

            # Search API next
            if fi is None:
                fi = search_month_contract(token, symbol, y, m)

            # BOD fallback (works mainly for current/upcoming listings)
            if fi is None and bod_uni.get(symbol):
                in_month = [x for x in bod_uni[symbol] if first <= x.expiry <= last]
                if in_month:
                    fi = min(in_month, key=lambda x: x.expiry)
                else:
                    def dist(x): return abs((x.expiry - first).days)
                    near = [x for x in bod_uni[symbol] if dist(x) <= 45]
                    fi = min(near or bod_uni[symbol], key=dist)

            if fi is None:
                print(f"[MISS] {symbol} {y}-{m:02d}: could not resolve instrument_key")
                continue

            print(f"Month {y}-{m:02d} -> {fi.tradingsymbol} ({fi.expiry})")

            for mins in plan.minutes_list:
                try:
                    df = fetch_month_candles_v3(token, fi, y, m, minutes=mins)
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

    print("\nDone. Re-run back-tests on the extended range when ready.")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fill continuous files via Upstox REST (multi-key fallback + expiry rules).")
    ap.add_argument("--start", default="2025-05-28", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (inclusive). Default: today")
    ap.add_argument("--minutes", nargs="+", type=int, default=[1, 3, 5], choices=[1, 3, 5], help="Timeframes to fetch")
    ap.add_argument("--sleep", type=float, default=0.7, help="Sleep seconds between API calls")
    args = ap.parse_args()
    run_gap_fill(Plan(start_ymd=args.start, end_ymd=args.end, minutes_list=tuple(args.minutes), sleep_sec=args.sleep))
