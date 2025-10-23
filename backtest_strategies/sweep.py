# sweep.py
from __future__ import annotations
import itertools
import subprocess
import sys
from pathlib import Path

PY = sys.executable
RUN = Path(__file__).parent / "quick_backtest.py"

def run(args):
    cmd = [PY, str(RUN)] + args
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout

def main():
    # Example sweep: RSI thresholds
    symbols = ["BANKNIFTY","NIFTY"]
    tfs = ["3m","5m"]
    entries = [58,60,62,64]
    exits  = [48,50,52,54]
    best = []
    for sym, tf, er, xr in itertools.product(symbols, tfs, entries, exits):
        out = run(["--symbol", sym, "--tf", tf, "--strategy","rsi_pullback",
                   "--entry_rsi", str(er), "--exit_rsi", str(xr),
                   "--cost","50","--slip","0.0"])
        # crude parse
        lines = out.splitlines()
        pnl_line = next((l for l in lines if "total_pnl" in l), "")
        pnl = float(pnl_line.split(":")[-1].strip()) if pnl_line else -9e9
        best.append((pnl, sym, tf, er, xr))
        print(f"{sym} {tf} er={er} xr={xr} pnl={pnl}")
    best.sort(reverse=True, key=lambda x: x[0])
    print("\nTop 10 by total_pnl:")
    for row in best[:10]:
        print(row)

if __name__ == "__main__":
    main()
