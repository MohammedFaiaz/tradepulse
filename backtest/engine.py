# backtest/engine.py
from __future__ import annotations
import math
from dataclasses import dataclass
import pandas as pd

@dataclass
class BTConfig:
    entry_rsi: float = 60.0
    exit_rsi: float = 50.0
    use_vwap_bias: bool = True
    use_cpr_bias: bool = True
    use_ema_bias: bool = True
    cost_per_trade: float = 50.0   # ₹ per roundtrip (both legs)
    slippage_points: float = 0.0   # add/subtract to fills
    squareoff_time: str = "15:15"

class Backtester:
    """
    Single-asset, long-only, intraday backtest on futures continuous series.
    Position: 0 or +1. Unit P&L is in price points; no lot sizing for now.
    """
    def __init__(self, cfg: BTConfig):
        self.cfg = cfg

    def _bias_long(self, row) -> bool:
        # Treat missing VWAP/CPR as neutral (do not block entries)
        vwap = row.get("vwap", pd.NA)
        tc   = row.get("tc", pd.NA)

        vwap_ok = True
        if self.cfg.use_vwap_bias and pd.notna(vwap):
            vwap_ok = (row["c"] > vwap)

        cpr_ok = True
        if self.cfg.use_cpr_bias and pd.notna(tc):
            cpr_ok = (row["c"] > tc)

        ema_ok = (not self.cfg.use_ema_bias) or (row["ema20"] > row["ema50"])
        return vwap_ok and cpr_ok and ema_ok

    def run(self, feats: pd.DataFrame) -> pd.DataFrame:
        """
        feats: DataFrame with columns c, rsi, [vwap], [tc], ema20, ema50
        Returns trades DataFrame with fills and P&L, plus an equity curve.
        """
        pos = 0
        entry_price = None
        trades = []
        sq_hour, sq_min = map(int, self.cfg.squareoff_time.split(":"))

        for t, row in feats.iterrows():
            price = float(row["c"])

            # Square-off guard
            if t.hour > sq_hour or (t.hour == sq_hour and t.minute >= sq_min):
                if pos == 1:
                    exit_px = price - self.cfg.slippage_points
                    trades.append((t, "EXIT_EOD", exit_px, entry_price))
                    pos = 0
                    entry_price = None
                continue

            bias_ok = self._bias_long(row)
            rsi = row.get("rsi", float("nan"))

            if pos == 0:
                # ENTRY: RSI above threshold + bias
                if (not math.isnan(rsi)) and (rsi > self.cfg.entry_rsi) and bias_ok:
                    entry_price = price + self.cfg.slippage_points
                    trades.append((t, "BUY", entry_price, None))
                    pos = 1
            else:
                # EXIT rules
                exit_now = False
                if (not math.isnan(rsi)) and (rsi < self.cfg.exit_rsi):
                    exit_now = True
                vwap = row.get("vwap", pd.NA)
                if pd.notna(vwap) and price < vwap:
                    exit_now = True

                if exit_now:
                    exit_px = price - self.cfg.slippage_points
                    trades.append((t, "SELL", exit_px, entry_price))
                    pos = 0
                    entry_price = None

        if pos == 1 and entry_price is not None:
            last_t = feats.index[-1]
            last_p = float(feats.iloc[-1]["c"])
            trades.append((last_t, "EXIT_CLOSE", last_p, entry_price))

        if not trades:
            return pd.DataFrame(columns=["t","side","price","ref","pnl","equity"])

        df = pd.DataFrame(trades, columns=["t","side","price","ref"])
        df["pnl"] = 0.0
        equity = []
        cum = 0.0

        open_px = None
        for i, r in df.iterrows():
            if r["side"] == "BUY":
                open_px = r["price"]
                df.at[i, "ref"] = None
            else:
                if open_px is None:
                    continue
                gross = r["price"] - open_px
                net = gross - self.cfg.cost_per_trade
                df.at[i, "ref"] = open_px
                df.at[i, "pnl"] = net
                cum += net
                open_px = None
            equity.append(cum)

        df["t"] = pd.to_datetime(df["t"])
        df = df.set_index("t")
        # For entries, equity repeats last cum
        if len(equity) < len(df):
            equity = equity + [cum] * (len(df) - len(equity))
        df["equity"] = equity
        return df
