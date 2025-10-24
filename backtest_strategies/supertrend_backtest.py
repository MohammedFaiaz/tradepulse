"""SuperTrend crossover backtest without external dependencies.

This module mirrors the signal logic described in ``STRATEGY_1_PINESCRIPT.txt``
using only the Python standard library so it can run in restricted
environments where ``numpy`` or ``pandas`` are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
from itertools import product
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Candle:
    """Single OHLCV candle."""

    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class PreparedRow:
    """Candle enriched with all indicator values used by the backtest."""

    candle: Candle
    supertrend: float
    final_upper: float
    final_lower: float
    direction: int
    atr_stop_band: float
    bull_signal: bool
    bear_signal: bool


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    gross: float
    exit_reason: str
    bars_held: int
    stop_level: float
    target_level: float
    quantity: int


@dataclass
class SupertrendConfig:
    data_dir: Path = Path("data/continuous")
    symbol: str = "BANKNIFTY"
    timeframe: str = "1m"
    supertrend_factor: float = 6.0
    supertrend_atr_length: int = 8
    atr_risk_length: int = 14
    atr_stop_multiplier: float = 3.0
    target_multiple: float = 1.4
    breakeven_multiple: float = 0.0
    cost_per_trade: float = 0.0
    lot_size: int = 30
    max_daily_loss: float = 5000.0
    max_trade_loss: float = 5000.0
    start: Optional[datetime] = None
    end: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


SYMBOL_LOT_SIZES: Dict[str, int] = {
    "BANKNIFTY": 30,
    "NIFTY": 75,
}


@dataclass(frozen=True)
class ParameterPreset:
    """Named set of tuning parameters for a symbol/timeframe combination."""

    name: str
    supertrend_factor: float
    supertrend_atr_length: int
    atr_stop_multiplier: float
    target_multiple: float
    breakeven_multiple: float = 0.0
    description: str = ""

    def apply(self, cfg: SupertrendConfig) -> SupertrendConfig:
        """Return ``cfg`` with the preset values baked in."""

        return replace(
            cfg,
            supertrend_factor=self.supertrend_factor,
            supertrend_atr_length=self.supertrend_atr_length,
            atr_stop_multiplier=self.atr_stop_multiplier,
            target_multiple=self.target_multiple,
            breakeven_multiple=self.breakeven_multiple,
        )


PRESET_LIBRARY: Dict[tuple, Dict[str, ParameterPreset]] = {
    ("BANKNIFTY", "5m"): {
        "default": ParameterPreset(
            name="default",
            supertrend_factor=3.5,
            supertrend_atr_length=10,
            atr_stop_multiplier=2.5,
            target_multiple=1.8,
            breakeven_multiple=0.0,
            description="Higher trade frequency tune (~280 trades, ~₹98k) with more heat",
        ),
        "swing": ParameterPreset(
            name="swing",
            supertrend_factor=4.0,
            supertrend_atr_length=10,
            atr_stop_multiplier=2.5,
            target_multiple=1.8,
            breakeven_multiple=0.0,
            description="Prior default (~250 trades, ~₹90k) for a smoother equity curve",
        ),
        "balanced": ParameterPreset(
            name="balanced",
            supertrend_factor=6.0,
            supertrend_atr_length=8,
            atr_stop_multiplier=3.0,
            target_multiple=1.4,
            breakeven_multiple=0.0,
            description="Swing-friendly profile (~150 trades, ~₹65k) with deeper stops",
        ),
        "aggressive": ParameterPreset(
            name="aggressive",
            supertrend_factor=4.0,
            supertrend_atr_length=10,
            atr_stop_multiplier=2.5,
            target_multiple=1.0,
            breakeven_multiple=0.0,
            description="Looser SuperTrend prioritising trade count (~250 trades, ~₹62k)",
        ),
    },
    ("BANKNIFTY", "3m"): {
        "default": ParameterPreset(
            name="default",
            supertrend_factor=4.0,
            supertrend_atr_length=8,
            atr_stop_multiplier=2.0,
            target_multiple=1.6,
            breakeven_multiple=0.0,
            description="High-velocity setup (~408 trades, ~₹1.48L Jun 2024–May 2025)",
        ),
        "precision": ParameterPreset(
            name="precision",
            supertrend_factor=4.0,
            supertrend_atr_length=8,
            atr_stop_multiplier=2.5,
            target_multiple=1.4,
            breakeven_multiple=0.0,
            description="Balanced R-multiple (~391 trades, ~₹95k) with tighter stops",
        ),
        "balanced": ParameterPreset(
            name="balanced",
            supertrend_factor=4.0,
            supertrend_atr_length=10,
            atr_stop_multiplier=2.5,
            target_multiple=1.8,
            breakeven_multiple=0.0,
            description="Longer ATR lookback (~377 trades, ~₹40k) for calmer swings",
        ),
    },
    ("NIFTY", "3m"): {
        "default": ParameterPreset(
            name="default",
            supertrend_factor=4.0,
            supertrend_atr_length=10,
            atr_stop_multiplier=2.0,
            target_multiple=2.0,
            breakeven_multiple=0.0,
            description="Optimised profile (~418 trades, ~₹36k Jun 2024–May 2025) prioritising risk-adjusted PnL",
        ),
        "momentum": ParameterPreset(
            name="momentum",
            supertrend_factor=4.0,
            supertrend_atr_length=10,
            atr_stop_multiplier=2.0,
            target_multiple=1.6,
            breakeven_multiple=0.0,
            description="Higher trade frequency variant (~419 trades, ~₹31.8k) with a nearer target",
        ),
        "balanced": ParameterPreset(
            name="balanced",
            supertrend_factor=4.0,
            supertrend_atr_length=10,
            atr_stop_multiplier=2.0,
            target_multiple=1.8,
            breakeven_multiple=0.0,
            description="Middle-ground target (~418 trades, ~₹31k) between default and momentum",
        ),
        "adaptive": ParameterPreset(
            name="adaptive",
            supertrend_factor=4.0,
            supertrend_atr_length=8,
            atr_stop_multiplier=2.0,
            target_multiple=1.8,
            breakeven_multiple=0.0,
            description="Shorter ATR lookback (~415 trades, ~₹24k) for choppier NIFTY sessions",
        ),
    },
}


AUTO_OPTIMIZE_PRESET_NAMES = {"best", "auto-best", "optimise", "optimize"}


def get_presets(symbol: str, timeframe: str) -> Dict[str, ParameterPreset]:
    """Return the preset map for ``symbol``/``timeframe`` if it exists."""

    return PRESET_LIBRARY.get((symbol.upper(), timeframe.lower()), {})


def print_preset_catalog(symbol: str, timeframe: str) -> None:
    """Pretty-print the presets available for a symbol/timeframe pair."""

    entries = get_presets(symbol, timeframe)
    symbol_key = symbol.upper()
    timeframe_key = timeframe.lower()

    if not entries:
        print(f"No presets registered for {symbol_key} {timeframe_key}.")
        if PRESET_LIBRARY:
            print("Known symbol/timeframe combinations:")
            for (sym, tf), presets in sorted(PRESET_LIBRARY.items()):
                names = ", ".join(sorted(presets.keys()))
                print(f"  {sym} {tf}: {names}")
        return

    print(f"Presets for {symbol_key} {timeframe_key}:")
    for name, preset in sorted(entries.items()):
        description = f" -- {preset.description}" if preset.description else ""
        print(
            "  {name}: factor={factor}, atr_len={atr}, stop={stop}, target={target}, breakeven={breakeven}{desc}".format(
                name=name,
                factor=preset.supertrend_factor,
                atr=preset.supertrend_atr_length,
                stop=preset.atr_stop_multiplier,
                target=preset.target_multiple,
                breakeven=preset.breakeven_multiple,
                desc=description,
            )
        )

    print(
        "  best: auto-optimise using the scan grid (respects --scan-* overrides and optimisation filters)"
    )


def _is_nan(value: float) -> bool:
    return value != value


def infer_lot_size(symbol: str, fallback: int) -> int:
    """Return the natural lot size for a symbol, defaulting to ``fallback``."""

    key = symbol.upper()
    return SYMBOL_LOT_SIZES.get(key, fallback)


def _equal(a: float, b: float) -> bool:
    if _is_nan(a) and _is_nan(b):
        return True
    if _is_nan(a) or _is_nan(b):
        return False
    tolerance = 1e-9 * max(1.0, abs(a), abs(b))
    return abs(a - b) <= tolerance


# ---------------------------------------------------------------------------
# Indicator calculations (RMA/ATR/SuperTrend)
# ---------------------------------------------------------------------------


def _rma(values: List[float], length: int) -> List[float]:
    if length <= 0:
        raise ValueError("length must be > 0")
    out = [math.nan] * len(values)
    if not values:
        return out

    if len(values) < length:
        return out

    initial_avg = sum(values[:length]) / float(length)
    out[length - 1] = initial_avg
    prev = initial_avg
    coeff = length - 1
    for idx in range(length, len(values)):
        val = values[idx]
        prev = (prev * coeff + val) / float(length)
        out[idx] = prev
    return out


def _true_range(candles: List[Candle]) -> List[float]:
    tr: List[float] = []
    prev_close: Optional[float] = None
    for candle in candles:
        high_low = candle.high - candle.low
        high_close = abs(candle.high - prev_close) if prev_close is not None else 0.0
        low_close = abs(candle.low - prev_close) if prev_close is not None else 0.0
        tr.append(max(high_low, high_close, low_close))
        prev_close = candle.close
    return tr


def compute_atr(candles: List[Candle], length: int) -> List[float]:
    tr = _true_range(candles)
    return _rma(tr, length)


def compute_supertrend(
    candles: List[Candle],
    factor: float,
    atr_length: int,
) -> List[dict]:
    atr_vals = compute_atr(candles, atr_length)
    upper_band = [candle.close + factor * atr for candle, atr in zip(candles, atr_vals)]
    lower_band = [candle.close - factor * atr for candle, atr in zip(candles, atr_vals)]

    final_upper: List[float] = [math.nan] * len(candles)
    final_lower: List[float] = [math.nan] * len(candles)
    supertrend: List[float] = [math.nan] * len(candles)
    direction: List[int] = [0] * len(candles)

    prev_supertrend = math.nan
    prev_upper = math.nan
    prev_lower = math.nan

    for idx, candle in enumerate(candles):
        upper = upper_band[idx]
        lower = lower_band[idx]

        if idx > 0:
            prev_price = candles[idx - 1].close
            if not _is_nan(prev_lower):
                if not (lower > prev_lower or prev_price < prev_lower):
                    lower = prev_lower
            if not _is_nan(prev_upper):
                if not (upper < prev_upper or prev_price > prev_upper):
                    upper = prev_upper

        prev_atr = atr_vals[idx - 1] if idx > 0 else math.nan
        if _is_nan(prev_atr):
            dir_val = 1
        elif _equal(prev_supertrend, prev_upper):
            dir_val = -1 if candle.close > upper else 1
        else:
            dir_val = 1 if candle.close < lower else -1

        st_val = lower if dir_val == -1 else upper

        final_upper[idx] = upper
        final_lower[idx] = lower
        supertrend[idx] = st_val
        direction[idx] = dir_val

        prev_supertrend = st_val
        prev_upper = upper
        prev_lower = lower

    result: List[dict] = []
    for idx in range(len(candles)):
        result.append(
            {
                "supertrend": supertrend[idx],
                "direction": direction[idx],
                "final_upper": final_upper[idx],
                "final_lower": final_lower[idx],
                "atr": atr_vals[idx],
            }
        )
    return result


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def load_ohlc(path: Path) -> List[Candle]:
    candles: List[Candle] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"t", "o", "h", "l", "c"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        for row in reader:
            time = datetime.fromisoformat(row["t"])
            candles.append(
                Candle(
                    time=time,
                    open=float(row["o"]),
                    high=float(row["h"]),
                    low=float(row["l"]),
                    close=float(row["c"]),
                )
            )
    candles.sort(key=lambda candle: candle.time)
    return candles


def prepare_dataset_from_candles(candles: List[Candle], cfg: SupertrendConfig) -> List[PreparedRow]:
    if not candles:
        return []

    filtered = candles
    if cfg.start is not None:
        filtered = [c for c in filtered if c.time >= cfg.start]
    if cfg.end is not None:
        filtered = [c for c in filtered if c.time <= cfg.end]

    if not filtered:
        return []

    indicator_rows = compute_supertrend(
        filtered,
        factor=cfg.supertrend_factor,
        atr_length=cfg.supertrend_atr_length,
    )

    atr_risk = compute_atr(filtered, cfg.atr_risk_length)

    prepared: List[PreparedRow] = []
    for idx, candle in enumerate(filtered):
        st_val = indicator_rows[idx]["supertrend"]
        atr_band = atr_risk[idx]
        if _is_nan(st_val) or _is_nan(atr_band):
            continue

        prev_close = filtered[idx - 1].close if idx > 0 else math.nan
        prev_supertrend = indicator_rows[idx - 1]["supertrend"] if idx > 0 else math.nan
        close_price = candle.close

        bull_signal = (
            not _is_nan(prev_supertrend)
            and close_price > st_val
            and (prev_close <= prev_supertrend if not _is_nan(prev_close) else False)
        )
        bear_signal = (
            not _is_nan(prev_supertrend)
            and close_price < st_val
            and (prev_close >= prev_supertrend if not _is_nan(prev_close) else False)
        )

        prepared.append(
            PreparedRow(
                candle=candle,
                supertrend=st_val,
                final_upper=indicator_rows[idx]["final_upper"],
                final_lower=indicator_rows[idx]["final_lower"],
                direction=indicator_rows[idx]["direction"],
                atr_stop_band=atr_band * cfg.atr_stop_multiplier,
                bull_signal=bull_signal,
                bear_signal=bear_signal,
            )
        )
    return prepared


def prepare_dataset(cfg: SupertrendConfig) -> List[PreparedRow]:
    file_path = cfg.data_dir / f"{cfg.symbol}_CONT_{cfg.timeframe}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    candles = load_ohlc(file_path)
    return prepare_dataset_from_candles(candles, cfg)


# ---------------------------------------------------------------------------
# Backtest execution
# ---------------------------------------------------------------------------


def _update_stop(side: str, current_stop: float, row: PreparedRow) -> float:
    band = row.atr_stop_band
    if side == "long":
        candidate = row.candle.low - band
        return max(current_stop, candidate)
    candidate = row.candle.high + band
    return min(current_stop, candidate)


def run_backtest(rows: List[PreparedRow], cfg: SupertrendConfig) -> List[Trade]:
    trades: List[Trade] = []
    position = None
    equity = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0

    daily_pnl: Dict[date, float] = {}
    daily_paused: Dict[date, bool] = {}
    prev_row: Optional[PreparedRow] = None
    current_day: Optional[date] = None

    def close_position(exit_price: float, exit_time: datetime, exit_reason: str) -> None:
        nonlocal position, equity, peak_equity, max_drawdown

        if position is None:
            return

        side = position["side"]
        direction = 1 if side == "long" else -1
        gross_points = (exit_price - position["entry_price"]) * direction
        gross = gross_points * cfg.lot_size
        pnl = gross - cfg.cost_per_trade

        equity += pnl
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        max_drawdown = max(max_drawdown, drawdown)

        trade = Trade(
            entry_time=position["entry_time"],
            exit_time=exit_time,
            side=side,
            entry_price=position["entry_price"],
            exit_price=exit_price,
            pnl=pnl,
            gross=gross,
            exit_reason=exit_reason,
            bars_held=position["bars_held"],
            stop_level=position["stop"],
            target_level=position["target"],
            quantity=cfg.lot_size,
        )
        trades.append(trade)

        exit_day = exit_time.date()
        if exit_day not in daily_pnl:
            daily_pnl[exit_day] = 0.0
        if exit_day not in daily_paused:
            daily_paused[exit_day] = False
        day_total = daily_pnl[exit_day] + pnl
        daily_pnl[exit_day] = day_total
        if day_total <= -cfg.max_daily_loss:
            daily_paused[exit_day] = True

        position = None

    for row in rows:
        candle = row.candle
        candle_day = candle.time.date()

        if current_day is None:
            current_day = candle_day
            daily_pnl.setdefault(current_day, 0.0)
            daily_paused.setdefault(current_day, False)
        elif candle_day != current_day:
            if position is not None and prev_row is not None:
                close_position(prev_row.candle.close, prev_row.candle.time, "EOD")
            current_day = candle_day
            daily_pnl.setdefault(current_day, 0.0)
            daily_paused.setdefault(current_day, False)

        if position is not None:
            position["bars_held"] += 1
            side = position["side"]
            position["stop"] = _update_stop(side, position["stop"], row)

            risk = position.get("risk", 0.0)
            if cfg.breakeven_multiple > 0 and risk > 0 and not position.get("breakeven_triggered", False):
                if side == "long":
                    trigger_price = position["entry_price"] + risk * cfg.breakeven_multiple
                    if candle.high >= trigger_price:
                        position["stop"] = max(position["stop"], position["entry_price"])
                        position["breakeven_triggered"] = True
                else:
                    trigger_price = position["entry_price"] - risk * cfg.breakeven_multiple
                    if candle.low <= trigger_price:
                        position["stop"] = min(position["stop"], position["entry_price"])
                        position["breakeven_triggered"] = True

            exit_price = None
            exit_reason = None

            if side == "long":
                if candle.low <= position["stop"]:
                    exit_price = position["stop"]
                    exit_reason = "STOP"
                elif candle.high >= position["target"]:
                    exit_price = position["target"]
                    exit_reason = "TARGET"
            else:
                if candle.high >= position["stop"]:
                    exit_price = position["stop"]
                    exit_reason = "STOP"
                elif candle.low <= position["target"]:
                    exit_price = position["target"]
                    exit_reason = "TARGET"

            if exit_price is None:
                if side == "long" and row.bear_signal:
                    exit_price = candle.close
                    exit_reason = "REVERSE"
                elif side == "short" and row.bull_signal:
                    exit_price = candle.close
                    exit_reason = "REVERSE"

            if exit_price is not None:
                close_position(exit_price, candle.time, exit_reason)
                prev_row = row
                continue

        if position is None:
            if daily_paused.get(current_day, False):
                prev_row = row
                continue

            if row.bull_signal:
                entry_price = candle.close
                stop = candle.low - row.atr_stop_band
                allowed_loss_value = cfg.max_daily_loss + daily_pnl[current_day]
                if allowed_loss_value <= 0:
                    prev_row = row
                    continue
                per_trade_cap = min(cfg.max_trade_loss, allowed_loss_value)
                if per_trade_cap <= 0:
                    prev_row = row
                    continue
                capped_loss_value = min((entry_price - stop) * cfg.lot_size, per_trade_cap)
                risk_cap_points = capped_loss_value / cfg.lot_size
                stop = max(stop, entry_price - risk_cap_points)
                risk = entry_price - stop
                risk_value = risk * cfg.lot_size
                if risk <= 0 or risk_value <= 1e-6:
                    prev_row = row
                    continue
                target = entry_price + risk * cfg.target_multiple if risk > 0 else entry_price
                position = {
                    "side": "long",
                    "entry_time": candle.time,
                    "entry_price": entry_price,
                    "stop": stop,
                    "target": target,
                    "bars_held": 0,
                    "risk": risk,
                    "risk_value": risk_value,
                    "breakeven_triggered": False,
                }
            elif row.bear_signal:
                entry_price = candle.close
                stop = candle.high + row.atr_stop_band
                allowed_loss_value = cfg.max_daily_loss + daily_pnl[current_day]
                if allowed_loss_value <= 0:
                    prev_row = row
                    continue
                per_trade_cap = min(cfg.max_trade_loss, allowed_loss_value)
                if per_trade_cap <= 0:
                    prev_row = row
                    continue
                capped_loss_value = min((stop - entry_price) * cfg.lot_size, per_trade_cap)
                risk_cap_points = capped_loss_value / cfg.lot_size
                stop = min(stop, entry_price + risk_cap_points)
                risk = stop - entry_price
                risk_value = risk * cfg.lot_size
                if risk <= 0 or risk_value <= 1e-6:
                    prev_row = row
                    continue
                target = entry_price - risk * cfg.target_multiple if risk > 0 else entry_price
                position = {
                    "side": "short",
                    "entry_time": candle.time,
                    "entry_price": entry_price,
                    "stop": stop,
                    "target": target,
                    "bars_held": 0,
                    "risk": risk,
                    "risk_value": risk_value,
                    "breakeven_triggered": False,
                }

        prev_row = row

    if position is not None and prev_row is not None:
        close_position(prev_row.candle.close, prev_row.candle.time, "FINAL")

    # annotate drawdown on list (using attribute on list via setattr is not possible)
    for trade in trades:
        setattr(trade, "max_drawdown", max_drawdown)

    return trades


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def summarize_trades(trades: List[Trade]) -> List[tuple]:
    if not trades:
        return [
            ("trades", 0),
            ("wins", 0),
            ("losses", 0),
            ("win_rate", 0.0),
            ("avg_pnl", 0.0),
            ("total_pnl", 0.0),
            ("expectancy", 0.0),
            ("max_dd", 0.0),
        ]

    wins = sum(1 for trade in trades if trade.pnl > 0)
    losses = sum(1 for trade in trades if trade.pnl < 0)
    total = len(trades)
    total_pnl = sum(trade.pnl for trade in trades)
    avg_pnl = total_pnl / total if total else 0.0
    win_rate = (wins / total) * 100 if total else 0.0
    max_dd = getattr(trades[0], "max_drawdown", 0.0)

    return [
        ("trades", total),
        ("wins", wins),
        ("losses", losses),
        ("win_rate", round(win_rate, 2)),
        ("avg_pnl", round(avg_pnl, 2)),
        ("total_pnl", round(total_pnl, 2)),
        ("expectancy", round(avg_pnl, 2)),
        ("max_dd", round(max_dd, 2)),
    ]


def print_trades(trades: List[Trade]) -> None:
    if not trades:
        print("No trades generated for the selected period.")
        return

    header = (
        "entry_time",
        "exit_time",
        "side",
        "qty",
        "entry",
        "exit",
        "pnl",
        "reason",
        "bars",
        "stop",
        "target",
    )
    print("Trades:")
    print(" | ".join(f"{h:>12}" for h in header))
    for trade in trades:
        print(
            " | ".join(
                [
                    trade.entry_time.isoformat(sep=" "),
                    trade.exit_time.isoformat(sep=" "),
                    f"{trade.side:>5}",
                    f"{trade.quantity:>5}",
                    f"{trade.entry_price:>8.2f}",
                    f"{trade.exit_price:>8.2f}",
                    f"{trade.pnl:>8.2f}",
                    f"{trade.exit_reason:>7}",
                    f"{trade.bars_held:>4}",
                    f"{trade.stop_level:>8.2f}",
                    f"{trade.target_level:>8.2f}",
                ]
            )
        )


def print_summary(summary: List[tuple]) -> None:
    print("\nSummary:")
    for metric, value in summary:
        print(f"{metric:>12}: {value}")


def print_sample_context(
    rows: List[PreparedRow], summary: Dict[str, float], cfg: SupertrendConfig
) -> None:
    """Provide extra context when a run spans only a handful of sessions."""

    if not rows:
        return

    session_dates = sorted({row.candle.time.date() for row in rows})
    if not session_dates:
        return

    first_session, last_session = session_dates[0], session_dates[-1]
    session_count = len(session_dates)

    print(
        "\nSample coverage: {sessions} session(s) from {first} to {last}.".format(
            sessions=session_count,
            first=first_session.isoformat(),
            last=last_session.isoformat(),
        )
    )

    trade_count = int(summary.get("trades", 0))
    if session_count <= 5 or trade_count <= 12:
        print(
            "Note: Only {trades} trades were generated across {sessions} session(s); "
            "consider widening --start/--end or relaxing optimisation filters for "
            "more robust statistics.".format(trades=trade_count, sessions=session_count)
        )

    if cfg.start and cfg.end:
        requested_days = (cfg.end.date() - cfg.start.date()).days + 1
        if requested_days > session_count:
            print(
                "Filtered trading days fall within a {days}-day calendar window; "
                "weekends or holidays may shrink the effective sample.".format(
                    days=requested_days
                )
            )


def print_line_continuation_hint(symbols: Iterable[str]) -> None:
    """Explain how to enter multi-argument commands on different shells."""

    symbols = list(symbols)
    if not symbols:
        return

    system = platform.system().lower()
    if "windows" in system:
        print(
            "\nPowerShell tip: use the backtick (`) for line continuation or run the "
            "command on a single line, e.g.\n"
            "  python -m backtest_strategies.supertrend_backtest --symbols "
            "{syms} --tf 3m --start YYYY-MM-DD --end YYYY-MM-DD".format(
                syms=",".join(symbols)
            )
        )
    else:
        # No-op on Unix shells where trailing backslashes already work as shown
        return


# ---------------------------------------------------------------------------
# Parameter scan helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanGrid:
    factors: Tuple[float, ...]
    atr_lengths: Tuple[int, ...]
    stop_multipliers: Tuple[float, ...]
    target_multiples: Tuple[float, ...]
    breakeven_multiples: Tuple[float, ...]


GLOBAL_SCAN_GRID = ScanGrid(
    factors=(4.0, 5.0, 6.0),
    atr_lengths=(8, 10),
    stop_multipliers=(2.0, 2.5, 3.0),
    target_multiples=(1.0, 1.2, 1.4, 1.6, 1.8, 2.0),
    breakeven_multiples=(0.0, 0.4, 0.6),
)


SCAN_GRID_OVERRIDES: Dict[tuple, ScanGrid] = {
    ("NIFTY", "3m"): ScanGrid(
        factors=(3.5, 4.0, 4.5),
        atr_lengths=(8, 10, 12),
        stop_multipliers=(1.8, 2.0, 2.2, 2.5),
        target_multiples=(1.4, 1.6, 1.8, 2.0, 2.2),
        breakeven_multiples=(0.0, 0.3, 0.5),
    ),
}


DEFAULT_SCAN_LIMIT = 10


def resolve_scan_grid(symbol: str, timeframe: str) -> ScanGrid:
    key = (symbol.upper(), timeframe.lower())
    return SCAN_GRID_OVERRIDES.get(key, GLOBAL_SCAN_GRID)


def _sort_scan_rows(results: List[dict], metric: str) -> List[dict]:
    """Return ``results`` sorted according to ``metric`` (descending)."""

    def sort_tuple(row: dict) -> tuple:
        if metric == "trades":
            return (row["trades"], row["total_pnl"])
        if metric == "win_rate":
            return (row["win_rate"], row["total_pnl"])
        if metric == "avg_pnl":
            return (row["avg_pnl"], row["total_pnl"])
        if metric == "expectancy":
            return (row["expectancy"], row["total_pnl"])
        if metric == "pnl_trades":
            return (row["total_pnl"], row["trades"])
        # default to total pnl
        return (row["total_pnl"], row["win_rate"])

    return sorted(results, key=sort_tuple, reverse=True)


def _parse_list(text: Optional[str], default: List[float], cast) -> List[float]:
    if text in (None, ""):
        return default

    values: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(cast(part))
    return values or default


def _compute_scan_results(
    base_cfg: SupertrendConfig,
    args: argparse.Namespace,
    *,
    min_trades_override: Optional[int] = None,
    min_pnl_override: Optional[float] = None,
) -> List[dict]:
    file_path = base_cfg.data_dir / f"{base_cfg.symbol}_CONT_{base_cfg.timeframe}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    candles = load_ohlc(file_path)

    scan_grid = resolve_scan_grid(base_cfg.symbol, base_cfg.timeframe)
    factors = _parse_list(args.scan_factors, list(scan_grid.factors), float)
    atr_lengths = _parse_list(args.scan_atr_lengths, list(scan_grid.atr_lengths), int)
    stop_multipliers = _parse_list(args.scan_stop_multipliers, list(scan_grid.stop_multipliers), float)
    target_multiples = _parse_list(args.scan_target_multiples, list(scan_grid.target_multiples), float)
    breakeven_multiples = _parse_list(args.scan_breakeven_multiples, list(scan_grid.breakeven_multiples), float)
    min_trades_arg = args.scan_min_trades or 0
    min_pnl_arg = args.scan_min_pnl if args.scan_min_pnl is not None else -float("inf")
    min_trades = min_trades_override if min_trades_override is not None else min_trades_arg
    min_pnl = min_pnl_override if min_pnl_override is not None else min_pnl_arg

    combos = list(product(factors, atr_lengths, stop_multipliers, target_multiples, breakeven_multiples))
    if not combos:
        raise ValueError("No parameter combinations to scan")

    print(f"Scanning {len(combos)} parameter combinations...")

    results: List[dict] = []
    for factor, atr_len, stop_mult, target_mult, be_mult in combos:
        cfg_variant = replace(
            base_cfg,
            supertrend_factor=float(factor),
            supertrend_atr_length=int(atr_len),
            atr_stop_multiplier=float(stop_mult),
            target_multiple=float(target_mult),
            breakeven_multiple=float(be_mult),
        )
        rows = prepare_dataset_from_candles(candles, cfg_variant)
        trades = run_backtest(rows, cfg_variant)
        summary = dict(summarize_trades(trades))
        total_pnl = float(summary["total_pnl"])
        trades_count = int(summary["trades"])

        if trades_count < min_trades or total_pnl < min_pnl:
            continue

        results.append(
            {
                "total_pnl": total_pnl,
                "win_rate": float(summary["win_rate"]),
                "trades": trades_count,
                "avg_pnl": float(summary["avg_pnl"]),
                "expectancy": float(summary["expectancy"]),
                "max_dd": float(summary["max_dd"]),
                "factor": float(factor),
                "atr_len": int(atr_len),
                "stop": float(stop_mult),
                "target": float(target_mult),
                "breakeven": float(be_mult),
            }
        )

    if not results:
        return []

    return results


def run_parameter_scan(base_cfg: SupertrendConfig, args: argparse.Namespace) -> List[dict]:
    results = _compute_scan_results(base_cfg, args)
    if not results:
        print("No parameter combinations matched the scan filters.")
        return []

    requested_limit = args.scan_limit
    sort_metric = args.scan_sort
    sorted_results = _sort_scan_rows(results, sort_metric)

    if requested_limit is None:
        limit = min(DEFAULT_SCAN_LIMIT, len(sorted_results))
    elif requested_limit <= 0:
        limit = len(sorted_results)
    else:
        limit = min(requested_limit, len(sorted_results))
    label = sort_metric.replace("_", " ")
    print(f"Top {limit} configurations by {label}:")
    for row in sorted_results[:limit]:
        print(
            "PNL={pnl:.2f}, win_rate={win:.2f}, trades={trades}, avg_pnl={avg:.2f}, expectancy={exp:.2f}, "
            "factor={factor}, atr_len={atr}, stop={stop}, target={target}, breakeven={be}".format(
                pnl=row["total_pnl"],
                win=row["win_rate"],
                trades=row["trades"],
                avg=row["avg_pnl"],
                exp=row["expectancy"],
                factor=row["factor"],
                atr=row["atr_len"],
                stop=row["stop"],
                target=row["target"],
                be=row["breakeven"],
            )
        )

    return sorted_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SuperTrend crossover backtest")
    parser.add_argument("--symbol", default="BANKNIFTY", help="Symbol prefix (e.g., BANKNIFTY or NIFTY)")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help=(
            "Comma-separated list of symbols to evaluate sequentially. When provided, the "
            "--symbol value is ignored and a combined summary is printed."
        ),
    )
    parser.add_argument("--tf", default="1m", help="Timeframe suffix used in the CSV file name")
    parser.add_argument("--data-dir", default="data/continuous", help="Directory containing *_CONT_*.csv files")
    parser.add_argument("--start", type=str, default=None, help="Optional start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Optional end date (YYYY-MM-DD)")
    parser.add_argument(
        "--preset",
        type=str,
        default="auto",
        help=(
            "Apply a named tuning preset before any explicit overrides. Use --list-presets "
            "to inspect the options for the requested symbol/timeframe (default: auto)"
        ),
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Display available presets for the provided symbol/timeframe and exit",
    )
    parser.add_argument(
        "--supertrend-factor",
        type=float,
        default=None,
        help="Multiplier applied to ATR when drawing the SuperTrend band (default: preset-specific)",
    )
    parser.add_argument(
        "--supertrend-atr-length",
        type=int,
        default=None,
        help="ATR length used inside the SuperTrend calculation (default: preset-specific)",
    )
    parser.add_argument(
        "--atr-risk-length",
        type=int,
        default=14,
        help="ATR length for risk/stop sizing (default: 14)",
    )
    parser.add_argument(
        "--atr-stop-multiplier",
        type=float,
        default=None,
        help="ATR multiple applied to the trailing stop (default: preset-specific)",
    )
    parser.add_argument(
        "--target-multiple",
        type=float,
        default=None,
        help="Risk multiple for the profit target (default: preset-specific)",
    )
    parser.add_argument(
        "--breakeven-multiple",
        type=float,
        default=None,
        help=(
            "Move the stop to breakeven once price moves this multiple of risk in the trade's favor"
            " (0 disables the behaviour; default: preset-specific)"
        ),
    )
    parser.add_argument("--cost", type=float, default=0.0, help="Per-trade transaction cost (deducted on exits)")
    parser.add_argument(
        "--lot-size",
        type=int,
        default=None,
        help="Number of units traded per signal (default: symbol specific, e.g. 30 for BANKNIFTY, 75 for NIFTY)",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=5000.0,
        help="Stop taking new trades once the day's realized PnL reaches this loss (default: 5000)",
    )
    parser.add_argument(
        "--max-trade-loss",
        type=float,
        default=None,
        help="Cap the worst-case loss per trade before costs; defaults to the daily loss cap",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Run a grid search across parameter lists instead of a single backtest",
    )
    parser.add_argument(
        "--scan-factors",
        type=str,
        default=None,
        help="Comma separated SuperTrend factors to test when --scan is used (default: 4,5,6)",
    )
    parser.add_argument(
        "--scan-atr-lengths",
        type=str,
        default=None,
        help="Comma separated ATR lengths for SuperTrend when --scan is used (default: 8,10)",
    )
    parser.add_argument(
        "--scan-stop-multipliers",
        type=str,
        default=None,
        help="Comma separated ATR stop multipliers when --scan is used (default: 2.0,2.5,3.0)",
    )
    parser.add_argument(
        "--scan-target-multiples",
        type=str,
        default=None,
        help=(
            "Comma separated risk multiples for profit targets when --scan is used "
            "(default: 1.0,1.2,1.4,1.6,1.8,2.0)"
        ),
    )
    parser.add_argument(
        "--scan-breakeven-multiples",
        type=str,
        default=None,
        help="Comma separated breakeven triggers when --scan is used (default: 0,0.4,0.6)",
    )
    parser.add_argument(
        "--scan-sort",
        choices=["pnl", "trades", "win_rate", "avg_pnl", "expectancy", "pnl_trades"],
        default="pnl",
        help=(
            "Metric used to sort scan results (default: pnl). Use 'pnl_trades' to prefer higher PnL with "
            "trade-count tie breaking."
        ),
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help="Maximum number of scan rows to display (default: 10)",
    )
    parser.add_argument(
        "--scan-min-trades",
        type=int,
        default=0,
        help="Discard scan combinations with fewer than this many trades",
    )
    parser.add_argument(
        "--scan-min-pnl",
        type=float,
        default=None,
        help="Discard scan combinations with total PnL below this threshold",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help=(
            "Automatically select the best parameter combination from the scan grid, apply it, and run the backtest"
        ),
    )
    parser.add_argument(
        "--optimize-sort",
        choices=["pnl", "trades", "win_rate", "avg_pnl", "expectancy", "pnl_trades"],
        default=None,
        help=(
            "Metric used when picking the optimisation winner (default: reuse --scan-sort, typically pnl)."
        ),
    )
    parser.add_argument(
        "--optimize-min-trades",
        type=int,
        default=None,
        help="Require at least this many trades when auto-selecting parameters",
    )
    parser.add_argument(
        "--optimize-min-pnl",
        type=float,
        default=None,
        help="Require at least this much total PnL when auto-selecting parameters",
    )
    return parser.parse_args(args=args)


def parse_optional_date(value: Optional[str], *, is_end: bool = False) -> Optional[datetime]:
    if value in (None, ""):
        return None

    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid date value '{value}': {exc}") from exc

    # Treat bare dates (YYYY-MM-DD) as whole-day ranges. Users typically pass the
    # Pine-style session dates without timestamps, so expand ``start`` to
    # midnight and ``end`` to the end of the day.
    is_date_only = len(text) == 10 and text.count("-") == 2 and "T" not in text and " " not in text
    if is_date_only:
        boundary = time.max if is_end else time.min
        return datetime.combine(parsed.date(), boundary)

    return parsed


def _run_single_symbol(
    args: argparse.Namespace,
    symbol: str,
    *,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Optional[Dict[str, float]]:
    symbol_key = symbol.upper()
    timeframe_key = args.tf.lower()

    if args.list_presets:
        print_preset_catalog(symbol_key, timeframe_key)
        return None

    inferred_lot_size = args.lot_size if args.lot_size is not None else infer_lot_size(symbol, 30)

    defaults = SupertrendConfig()
    per_trade_cap = args.max_trade_loss if args.max_trade_loss is not None else args.max_daily_loss
    cfg = SupertrendConfig(
        data_dir=Path(args.data_dir),
        symbol=symbol,
        timeframe=args.tf,
        supertrend_factor=defaults.supertrend_factor,
        supertrend_atr_length=defaults.supertrend_atr_length,
        atr_risk_length=args.atr_risk_length,
        atr_stop_multiplier=defaults.atr_stop_multiplier,
        target_multiple=defaults.target_multiple,
        breakeven_multiple=defaults.breakeven_multiple,
        cost_per_trade=args.cost,
        lot_size=inferred_lot_size,
        max_daily_loss=args.max_daily_loss,
        max_trade_loss=per_trade_cap,
        start=start,
        end=end,
    )

    preset_map = get_presets(symbol_key, timeframe_key)
    requested_preset = (args.preset or "auto").lower()
    applied_preset: Optional[str] = None
    scan_results: Optional[List[dict]] = None
    auto_metric: Optional[str] = None

    if requested_preset in AUTO_OPTIMIZE_PRESET_NAMES:
        scan_results = _compute_scan_results(cfg, args)
        if not scan_results:
            print(
                "No parameter combinations matched the optimisation filters for the 'best' preset."
            )
            return None

        auto_metric = args.optimize_sort or args.scan_sort
        ranked = _sort_scan_rows(scan_results, auto_metric)
        winner = ranked[0]
        print(
            "Preset '{name}' auto-selected parameters ({metric}) -> factor={factor}, atr_len={atr}, "
            "stop={stop}, target={target}, breakeven={be}, trades={trades}, pnl={pnl:.2f}".format(
                name=requested_preset,
                metric=(auto_metric or "pnl").replace("_", " "),
                factor=winner["factor"],
                atr=winner["atr_len"],
                stop=winner["stop"],
                target=winner["target"],
                be=winner["breakeven"],
                trades=winner["trades"],
                pnl=winner["total_pnl"],
            )
        )
        cfg = replace(
            cfg,
            supertrend_factor=winner["factor"],
            supertrend_atr_length=winner["atr_len"],
            atr_stop_multiplier=winner["stop"],
            target_multiple=winner["target"],
            breakeven_multiple=winner["breakeven"],
        )
        applied_preset = requested_preset
    elif requested_preset not in {"manual", "none"}:
        preset_obj: Optional[ParameterPreset]
        if requested_preset == "auto":
            preset_obj = preset_map.get("default")
            applied_preset = "default" if preset_obj else None
        else:
            preset_obj = preset_map.get(requested_preset)
            if preset_obj is None:
                raise SystemExit(
                    f"Unknown preset '{args.preset}' for {symbol_key} {timeframe_key}. "
                    "Run with --list-presets to inspect available names."
                )
            applied_preset = requested_preset

        if preset_obj is not None:
            cfg = preset_obj.apply(cfg)

    if args.supertrend_factor is not None:
        cfg = replace(cfg, supertrend_factor=args.supertrend_factor)
    if args.supertrend_atr_length is not None:
        cfg = replace(cfg, supertrend_atr_length=args.supertrend_atr_length)
    if args.atr_stop_multiplier is not None:
        cfg = replace(cfg, atr_stop_multiplier=args.atr_stop_multiplier)
    if args.target_multiple is not None:
        cfg = replace(cfg, target_multiple=args.target_multiple)
    if args.breakeven_multiple is not None:
        cfg = replace(cfg, breakeven_multiple=args.breakeven_multiple)

    if applied_preset and applied_preset not in AUTO_OPTIMIZE_PRESET_NAMES:
        preset_details = preset_map[applied_preset]
        print(
            "Using preset '{name}' for {symbol} {tf}: factor={factor}, atr_len={atr}, stop={stop}, "
            "target={target}, breakeven={breakeven}".format(
                name=applied_preset,
                symbol=symbol_key,
                tf=timeframe_key,
                factor=preset_details.supertrend_factor,
                atr=preset_details.supertrend_atr_length,
                stop=preset_details.atr_stop_multiplier,
                target=preset_details.target_multiple,
                breakeven=preset_details.breakeven_multiple,
            )
        )
    elif requested_preset == "auto" and not preset_map:
        print(f"No presets registered for {symbol_key} {timeframe_key}; using base defaults.")

    if args.scan:
        scan_results = run_parameter_scan(cfg, args)
        if not args.optimize:
            return None
    elif args.optimize:
        if scan_results is None:
            scan_results = _compute_scan_results(cfg, args)

    if args.optimize:
        if not scan_results:
            print("No parameter combinations matched the optimisation filters.")
            return None

        filtered = scan_results
        if args.optimize_min_trades is not None:
            filtered = [row for row in filtered if row["trades"] >= args.optimize_min_trades]
        if args.optimize_min_pnl is not None:
            filtered = [row for row in filtered if row["total_pnl"] >= args.optimize_min_pnl]

        if not filtered:
            print("No parameter combinations satisfied the optimisation thresholds.")
            return None

        optimize_metric = args.optimize_sort or args.scan_sort
        ranked = _sort_scan_rows(filtered, optimize_metric)
        winner = ranked[0]
        print(
            "Auto-selected parameters ({metric}) -> factor={factor}, atr_len={atr}, stop={stop}, "
            "target={target}, breakeven={be}, trades={trades}, pnl={pnl:.2f}".format(
                metric=optimize_metric.replace("_", " "),
                factor=winner["factor"],
                atr=winner["atr_len"],
                stop=winner["stop"],
                target=winner["target"],
                be=winner["breakeven"],
                trades=winner["trades"],
                pnl=winner["total_pnl"],
            )
        )
        cfg = replace(
            cfg,
            supertrend_factor=winner["factor"],
            supertrend_atr_length=winner["atr_len"],
            atr_stop_multiplier=winner["stop"],
            target_multiple=winner["target"],
            breakeven_multiple=winner["breakeven"],
        )

    rows = prepare_dataset(cfg)
    trades = run_backtest(rows, cfg)
    summary = summarize_trades(trades)
    summary_dict = dict(summary)

    print_trades(trades)
    print_summary(summary)
    print_sample_context(rows, summary_dict, cfg)

    summary_dict["symbol"] = symbol_key
    summary_dict["timeframe"] = timeframe_key
    return summary_dict


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = parse_args(cli_args)
    try:
        start = parse_optional_date(args.start)
        end = parse_optional_date(args.end, is_end=True)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.symbols:
        symbols = [token.strip() for token in args.symbols.split(",") if token.strip()]
        if not symbols:
            symbols = [args.symbol]
    else:
        symbols = [args.symbol]

    symbols = [sym.upper() for sym in symbols]

    print_line_continuation_hint(symbols)

    if args.list_presets:
        for idx, sym in enumerate(symbols):
            if idx:
                print()
            print_preset_catalog(sym, args.tf.lower())
        return

    aggregated: List[Dict[str, float]] = []
    for sym in symbols:
        symbol_args = argparse.Namespace(**vars(args))
        symbol_args.symbol = sym
        result = _run_single_symbol(symbol_args, sym, start=start, end=end)
        if result:
            aggregated.append(result)

    if len(aggregated) > 1:
        total_trades = sum(int(summary.get("trades", 0)) for summary in aggregated)
        total_wins = sum(int(summary.get("wins", 0)) for summary in aggregated)
        total_losses = sum(int(summary.get("losses", 0)) for summary in aggregated)
        total_pnl = sum(float(summary.get("total_pnl", 0.0)) for summary in aggregated)
        max_dd = max((float(summary.get("max_dd", 0.0)) for summary in aggregated), default=0.0)
        avg_pnl = total_pnl / total_trades if total_trades else 0.0
        win_rate = (total_wins / total_trades * 100.0) if total_trades else 0.0

        print("\nCombined summary across symbols:")
        print(f"      trades: {total_trades}")
        print(f"        wins: {total_wins}")
        print(f"      losses: {total_losses}")
        print(f"    win_rate: {win_rate:.2f}")
        print(f"     avg_pnl: {avg_pnl:.2f}")
        print(f"   total_pnl: {total_pnl:.2f}")
        print(f"  expectancy: {avg_pnl:.2f}")
        print(f"      max_dd: {max_dd:.2f}")

        best_symbol = max(aggregated, key=lambda summary: float(summary.get("total_pnl", 0.0)))
        print(
            "Top performer: {symbol} {tf} with total PnL {pnl:.2f} over {trades} trades.".format(
                symbol=best_symbol.get("symbol", ""),
                tf=best_symbol.get("timeframe", ""),
                pnl=float(best_symbol.get("total_pnl", 0.0)),
                trades=int(best_symbol.get("trades", 0)),
            )
        )


if __name__ == "__main__":
    main()

