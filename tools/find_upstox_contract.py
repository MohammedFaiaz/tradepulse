# tools/find_upstox_contract.py
from __future__ import annotations
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any
import requests

SECRETS = Path(r"C:\tradePulse\.secrets\token.json")
API_BASE = "https://api.upstox.com"

def load_token() -> str:
    obj = json.loads(SECRETS.read_text(encoding="utf-8"))
    tok = obj.get("access_token") or obj.get("accessToken") or obj.get("token")
    if not tok:
        raise RuntimeError("No access_token in .secrets\\token.json")
    return tok

def month_code(m:int)->str:
    return ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"][m-1]

def month_start_end(y:int,m:int):
    from datetime import timedelta
    first = date(y,m,1)
    last  = date(y+1,1,1) - timedelta(days=1) if m==12 else date(y,m+1,1) - timedelta(days=1)
    return first,last

def auth_get(url:str, token:str, params:Dict[str,Any]) -> Dict[str,Any]:
    r = requests.get(url, headers={"Authorization":f"Bearer {token}","Accept":"application/json"}, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def main(symbol:str, ym:str):
    token = load_token()
    y,m = map(int, ym.split("-"))
    mon = month_code(m)
    first,last = month_start_end(y,m)

    # try a basket of queries that often match monthly futures
    queries = [
        f"{symbol} FUT {mon} {y%100}",
        f"{symbol}{y%100}{mon}FUT",
        f"{symbol} {mon} {y%100} FUT",
        f"{symbol} FUT {y%100} {mon}",
        f"{symbol} {y%100}{mon} FUT",
        f"{symbol} {mon} FUT",
        f"{symbol} FUT {mon}",
    ]
    seen=set()
    rows=[]
    for q in queries:
        try:
            data = auth_get(f"{API_BASE}/v2/instruments/search", token, {"query": q, "segment": "NSE_FO"})
        except Exception as e:
            print(f"[ERR] query={q}: {e}")
            continue
        payload = data.get("data", [])
        for e in payload:
            ts = str(e.get("tradingsymbol") or e.get("trading_symbol") or e.get("symbol") or "")
            if not ts or ts in seen: 
                continue
            ul = str(e.get("underlying_symbol") or e.get("underlyingSymbol") or e.get("underlying") or "")
            typ= str(e.get("instrument_type") or e.get("instrumentType") or "")
            if ul != symbol or not ("FUT" in ts or typ.upper().startswith("FUT")):
                continue
            exp = e.get("expiry") or e.get("expiry_date")
            if exp is None:
                continue
            # expiry may be ISO or epoch-ms
            if isinstance(exp,(int,float)):
                from datetime import datetime
                expd = datetime.utcfromtimestamp(float(exp)/1000.0).date()
            else:
                try: expd = datetime.fromisoformat(str(exp)).date()
                except: continue
            ik = str(e.get("instrument_key") or e.get("instrumentKey") or e.get("token") or "")
            rows.append((ts, ik, expd))
            seen.add(ts)

    if not rows:
        print(f"[MISS] No candidates found for {symbol} {ym}")
        return

    # show table with markers for in-month
    rows.sort(key=lambda x: x[2])
    print(f"\nCandidates for {symbol} {ym} (expiry inside month marked '*'):")
    for ts, ik, expd in rows:
        mark = "*" if first <= expd <= last else " "
        print(f"{mark} {expd}  {ts}  {ik}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=["BANKNIFTY","NIFTY"])
    ap.add_argument("--ym", required=True, help="YYYY-MM (e.g. 2025-06)")
    args = ap.parse_args()
    main(args.symbol, args.ym)
