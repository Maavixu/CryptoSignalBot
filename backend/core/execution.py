"""
execution.py — Simulated order execution with slippage guard.
In paper mode: simulates limit fills with realistic latency + slippage.
In live mode (future): replaces simulation with real exchange calls.

Responsibilities:
  - Simulate/place limit orders
  - Monitor slippage (abort if > max_slippage_pct)
  - Create trade records and persist via storage
  - Update bankroll after each trade closes
  - Call Step 5 post-mortem on close
"""
import time
import uuid
import random
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .config import load_config, set_value
from .kill_switch import check as kill_switch_check
from .logger import get_logger
from .risk import RiskValidationResult, PositionSize
from .storage import (
    append_trade, update_trade, get_open_trades,
    get_trade_by_id, append_training_sample,
)

_log = get_logger("risk")  # execution logs under risk step


@dataclass
class OrderResult:
    success: bool
    trade_id: Optional[str]
    fill_price: float
    slippage_pct: float
    reason: str


async def execute_trade(
    symbol: str,
    direction: str,
    risk_result: RiskValidationResult,
    predicted_probability: float,
    feature_vector: Optional[list] = None,
    prediction_id: Optional[str] = None,
) -> OrderResult:
    """
    Execute a validated trade signal.
    Simulates a limit order with realistic slippage.
    Aborts if slippage exceeds configured threshold.
    """
    kill_switch_check()   # final check before any order

    cfg = load_config()
    pos: PositionSize = risk_result.position
    max_slippage = cfg["risk"]["max_slippage_pct"] / 100.0
    mode = cfg["bot"]["mode"]

    _log.info(
        f"Executing {direction} order: {symbol} "
        f"${pos.usd_size:.2f} @ {pos.entry_price:.4f}",
        data={
            "symbol": symbol, "direction": direction,
            "entry_price": pos.entry_price, "usd_size": pos.usd_size,
            "tp": pos.tp_price, "sl": pos.sl_price, "mode": mode,
        },
    )

    # ── Simulate limit order fill ─────────────────────────────────────────
    fill_price, slippage_pct = await _simulate_fill(
        entry_price=pos.entry_price,
        direction=direction,
        mode=mode,
    )

    # ── Slippage guard ────────────────────────────────────────────────────
    if abs(slippage_pct) > max_slippage * 100:
        reason = (
            f"Slippage abort: {slippage_pct:+.2f}% > max {max_slippage*100:.1f}%"
        )
        _log.warning(reason, data={"slippage_pct": slippage_pct, "symbol": symbol})
        return OrderResult(
            success=False, trade_id=None,
            fill_price=fill_price, slippage_pct=slippage_pct,
            reason=reason,
        )

    # ── Adjust TP/SL for actual fill price ────────────────────────────────
    sl_pct = cfg["trade"]["default_sl_pct"] / 100.0
    tp_pct = cfg["trade"]["default_tp_pct"] / 100.0

    if direction == "BUY":
        tp_price = fill_price * (1.0 + tp_pct)
        sl_price = fill_price * (1.0 - sl_pct)
    else:
        tp_price = fill_price * (1.0 - tp_pct)
        sl_price = fill_price * (1.0 + sl_pct)

    # ── Create trade record ───────────────────────────────────────────────
    trade_id = f"trade_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    trade = {
        "id": trade_id,
        "symbol": symbol,
        "direction": direction,
        "status": "open",
        "entry_price": round(fill_price, 6),
        "tp_price": round(tp_price, 6),
        "sl_price": round(sl_price, 6),
        "position_size_usd": pos.usd_size,
        "position_units": pos.units,
        "predicted_probability": predicted_probability,
        "kelly_fraction": pos.fraction,
        "reward_risk_ratio": pos.reward_risk_ratio,
        "slippage_pct": round(slippage_pct, 4),
        "feature_vector": feature_vector,
        "prediction_id": prediction_id,
        "outcome": None,
        "exit_price": None,
        "pnl_usd": None,
        "pnl_pct": None,
        "closed_at": None,
        "close_reason": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    append_trade(trade)

    _log.info(
        f"Trade opened: {trade_id} | {symbol} {direction} "
        f"@ {fill_price:.4f} | TP={tp_price:.4f} SL={sl_price:.4f}",
        data={
            "trade_id": trade_id, "symbol": symbol,
            "fill_price": fill_price, "tp_price": tp_price, "sl_price": sl_price,
            "slippage_pct": slippage_pct,
        },
    )

    return OrderResult(
        success=True,
        trade_id=trade_id,
        fill_price=fill_price,
        slippage_pct=slippage_pct,
        reason="Order filled successfully",
    )


async def close_trade(
    trade_id: str,
    current_price: float,
    close_reason: str,   # "tp_hit" | "sl_hit" | "manual" | "time_exit"
) -> Optional[dict]:
    """
    Close an open trade, compute PnL, update bankroll, trigger post-mortem.
    Returns the updated trade record or None if not found.
    """
    trade = get_trade_by_id(trade_id)
    if not trade or trade.get("status") != "open":
        _log.warning(f"close_trade: trade {trade_id} not found or already closed")
        return None

    direction    = trade["direction"]
    entry_price  = trade["entry_price"]
    position_usd = trade["position_size_usd"]

    # PnL calculation
    if direction == "BUY":
        pnl_pct = (current_price - entry_price) / entry_price * 100.0
    else:  # SELL/SHORT
        pnl_pct = (entry_price - current_price) / entry_price * 100.0

    pnl_usd = position_usd * (pnl_pct / 100.0)
    outcome = "win" if pnl_usd > 0 else "loss"

    # Determine trade age
    try:
        opened = datetime.fromisoformat(trade["created_at"][:19])
        duration_s = (datetime.utcnow() - opened).total_seconds()
    except Exception:
        duration_s = 0.0

    updates = {
        "status": "closed",
        "outcome": outcome,
        "exit_price": round(current_price, 6),
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 4),
        "close_reason": close_reason,
        "closed_at": datetime.utcnow().isoformat(),
        "duration_seconds": round(duration_s, 0),
    }
    update_trade(trade_id, updates)
    trade.update(updates)

    # Update bankroll
    cfg = load_config()
    new_bankroll = cfg["bankroll"]["current_usd"] + pnl_usd
    set_value("bankroll.current_usd", round(new_bankroll, 2))

    _log.info(
        f"Trade closed: {trade_id} | {outcome.upper()} "
        f"${pnl_usd:+.2f} ({pnl_pct:+.2f}%) via {close_reason}",
        data={
            "trade_id": trade_id, "symbol": trade["symbol"],
            "outcome": outcome, "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct, 4), "close_reason": close_reason,
            "new_bankroll": round(new_bankroll, 2),
        },
    )

    # Append training sample for model retraining (Step 5 feed)
    if trade.get("feature_vector"):
        binary_outcome = 1 if outcome == "win" else 0
        append_training_sample(
            features=trade["feature_vector"],
            outcome=binary_outcome,
            trade_id=trade_id,
        )
        _log.info(
            f"Training sample appended for {trade['symbol']} (outcome={binary_outcome})"
        )

    # Trigger Step 5 post-mortem
    _run_postmortem(trade)

    return trade


def check_open_trades_prices(current_prices: dict) -> list:
    """
    Check all open trades against current prices.
    Closes trades where TP or SL has been hit.
    Returns list of closed trade records.
    Called by the background scheduler every N seconds.
    """
    open_trades = get_open_trades()
    if not open_trades:
        return []

    closed = []
    for trade in open_trades:
        symbol = trade["symbol"]
        price = current_prices.get(symbol)
        if price is None:
            continue

        tp = trade["tp_price"]
        sl = trade["sl_price"]
        direction = trade["direction"]

        hit_tp = (direction == "BUY"  and price >= tp) or \
                 (direction == "SELL" and price <= tp)
        hit_sl = (direction == "BUY"  and price <= sl) or \
                 (direction == "SELL" and price >= sl)

        if hit_tp or hit_sl:
            reason = "tp_hit" if hit_tp else "sl_hit"
            exit_price = tp if hit_tp else sl

            # Run close synchronously (scheduler context)
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                close_trade(trade["id"], exit_price, reason)
            )
            loop.close()
            if result:
                closed.append(result)

    return closed


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _simulate_fill(
    entry_price: float,
    direction: str,
    mode: str,
) -> tuple:
    """
    Simulate realistic limit order fill.
    Adds small random slippage to model real-world execution.
    """
    # Tiny delay simulating network round-trip
    await asyncio.sleep(random.uniform(0.05, 0.3))

    # Slippage: normally distributed, mean 0, std 0.1%, capped at ±0.5%
    slippage_pct = random.gauss(0.0, 0.10)
    slippage_pct = max(min(slippage_pct, 0.50), -0.50)

    # Adverse fill: buy fills slightly higher, sell slightly lower
    if direction == "BUY":
        fill_multiplier = 1.0 + abs(slippage_pct) / 100.0
    else:
        fill_multiplier = 1.0 - abs(slippage_pct) / 100.0

    fill_price = entry_price * fill_multiplier

    return round(fill_price, 6), round(slippage_pct, 4)


def _run_postmortem(trade: dict):
    """Trigger Step 5 post-mortem analysis on a closed trade."""
    try:
        from .compound import run_postmortem
        run_postmortem(trade)
    except Exception as e:
        _log.error(f"Post-mortem failed for {trade.get('id')}: {e}")
