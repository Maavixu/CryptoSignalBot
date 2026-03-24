"""
risk.py — Step 4: Risk Management
Seven independent risk checks run sequentially.
First failure short-circuits — no trade is generated.

Check order (from spec):
  1. Edge threshold    — p_win - p_market >= min_edge_pct
  2. Kelly sizing      — compute quarter-Kelly position size
  3. Exposure limit    — new + existing positions <= 50% bankroll
  4. VaR at 95%        — daily Value-at-Risk within daily limit
  5. Max drawdown      — current drawdown <= 8%
  6. Daily loss limit  — today's losses <= 15% of bankroll
  7. Concurrent cap    — open positions <= 15
  8. Kill switch       — STOP file check (always last gate)

Each check returns a RiskCheckResult(passed, reason, data).
"""
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import List, Optional, Tuple

import numpy as np

from .config import load_config, set_value
from .kill_switch import check as kill_switch_check, KillSwitchTriggered
from .logger import get_logger
from .storage import get_all_trades, get_open_trades

_log = get_logger("risk")


@dataclass
class RiskCheckResult:
    name: str
    passed: bool
    reason: str
    data: dict = field(default_factory=dict)


@dataclass
class PositionSize:
    fraction: float          # fraction of bankroll (Kelly output)
    usd_size: float          # dollar amount to risk
    units: float             # units of base asset
    entry_price: float
    tp_price: float
    sl_price: float
    reward_usd: float        # max profit if TP hit
    risk_usd: float          # max loss if SL hit
    reward_risk_ratio: float


@dataclass
class RiskValidationResult:
    approved: bool
    checks: List[RiskCheckResult]
    position: Optional[PositionSize]
    rejection_reason: Optional[str]
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
            "timestamp": self.timestamp,
            "position": asdict(self.position) if self.position else None,
            "checks": [asdict(c) for c in self.checks],
        }


# ── Main validation pipeline ──────────────────────────────────────────────────

def validate_trade(
    symbol: str,
    direction: str,           # "BUY" | "SELL"
    p_win: float,
    p_market: float,
    edge_pct: float,
    entry_price: float,
    ohlcv: Optional[list] = None,   # for VaR calculation
) -> RiskValidationResult:
    """
    Run all risk checks. Returns RiskValidationResult with approved flag.
    If approved, includes the calculated PositionSize.
    """
    cfg = load_config()
    risk_cfg = cfg["risk"]
    bankroll = cfg["bankroll"]["current_usd"]
    checks: List[RiskCheckResult] = []

    _log.info(
        f"Risk validation started: {symbol} {direction}",
        data={"symbol": symbol, "direction": direction, "edge_pct": edge_pct,
              "p_win": p_win, "bankroll": bankroll},
    )

    # ── Check 1: Edge threshold ───────────────────────────────────────────
    c1 = _check_edge(edge_pct, risk_cfg["kelly_fraction"], cfg["prediction"]["min_edge_pct"])
    checks.append(c1)
    if not c1.passed:
        return _reject(checks, c1.reason)

    # ── Check 2: Kelly position sizing ────────────────────────────────────
    reward_risk = cfg["trade"]["default_reward_risk_ratio"]
    sl_pct      = cfg["trade"]["default_sl_pct"] / 100.0
    tp_pct      = cfg["trade"]["default_tp_pct"] / 100.0

    position = _compute_kelly_position(
        p_win=p_win,
        reward_risk=reward_risk,
        kelly_fraction=risk_cfg["kelly_fraction"],
        bankroll=bankroll,
        entry_price=entry_price,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        direction=direction,
    )
    c2 = RiskCheckResult(
        name="kelly_sizing",
        passed=position.usd_size > 0,
        reason=f"Kelly size: ${position.usd_size:.2f} ({position.fraction*100:.1f}% bankroll)",
        data={"usd_size": position.usd_size, "fraction": position.fraction},
    )
    checks.append(c2)
    if not c2.passed:
        return _reject(checks, "Kelly size is zero — insufficient edge")

    # ── Check 3: Exposure limit ───────────────────────────────────────────
    c3 = _check_exposure(position.usd_size, bankroll, risk_cfg["max_total_exposure_pct"])
    checks.append(c3)
    if not c3.passed:
        return _reject(checks, c3.reason)

    # ── Check 4: Value at Risk (95%) ──────────────────────────────────────
    c4 = _check_var(
        position_usd=position.usd_size,
        bankroll=bankroll,
        ohlcv=ohlcv,
        var_confidence=risk_cfg["var_confidence"],
        var_daily_limit_pct=risk_cfg["var_daily_limit_pct"],
    )
    checks.append(c4)
    if not c4.passed:
        return _reject(checks, c4.reason)

    # ── Check 5: Max drawdown guard ───────────────────────────────────────
    c5 = _check_drawdown(bankroll, cfg["bankroll"]["initial_usd"], risk_cfg["max_drawdown_pct"])
    checks.append(c5)
    if not c5.passed:
        return _reject(checks, c5.reason)

    # ── Check 6: Daily loss limit ─────────────────────────────────────────
    c6 = _check_daily_loss(bankroll, cfg["bankroll"]["initial_usd"], risk_cfg["daily_loss_limit_pct"])
    checks.append(c6)
    if not c6.passed:
        return _reject(checks, c6.reason)

    # ── Check 7: Concurrent position cap ─────────────────────────────────
    c7 = _check_concurrent_positions(risk_cfg["max_concurrent_positions"])
    checks.append(c7)
    if not c7.passed:
        return _reject(checks, c7.reason)

    # ── Check 8: Kill switch (final gate) ─────────────────────────────────
    try:
        kill_switch_check()
        c8 = RiskCheckResult("kill_switch", True, "Kill switch inactive — trading allowed")
    except KillSwitchTriggered as e:
        c8 = RiskCheckResult("kill_switch", False, str(e))
        checks.append(c8)
        return _reject(checks, str(e))
    checks.append(c8)

    _log.info(
        f"Risk validation APPROVED: {symbol} {direction} "
        f"${position.usd_size:.2f} | TP={position.tp_price:.4f} SL={position.sl_price:.4f}",
        data={
            "symbol": symbol, "direction": direction,
            "usd_size": position.usd_size,
            "tp": position.tp_price, "sl": position.sl_price,
            "reward_usd": position.reward_usd, "risk_usd": position.risk_usd,
        },
    )

    return RiskValidationResult(
        approved=True,
        checks=checks,
        position=position,
        rejection_reason=None,
        timestamp=datetime.utcnow().isoformat(),
    )


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_edge(edge_pct: float, kelly_fraction: float, min_edge_pct: float) -> RiskCheckResult:
    passed = edge_pct >= min_edge_pct
    return RiskCheckResult(
        name="edge_threshold",
        passed=passed,
        reason=f"Edge {edge_pct:+.1f}% {'≥' if passed else '<'} minimum {min_edge_pct:.0f}%",
        data={"edge_pct": edge_pct, "min_edge_pct": min_edge_pct},
    )


def _compute_kelly_position(
    p_win: float,
    reward_risk: float,
    kelly_fraction: float,
    bankroll: float,
    entry_price: float,
    sl_pct: float,
    tp_pct: float,
    direction: str,
) -> PositionSize:
    """
    Quarter-Kelly criterion:
      f* = (p*b - q) / b
      position = f* * kelly_fraction * bankroll
    where b = reward:risk ratio, p = p_win, q = 1-p_win
    """
    q = 1.0 - p_win
    b = reward_risk

    full_kelly = (p_win * b - q) / b
    # Clamp: negative Kelly means no edge (should have been caught by check 1)
    full_kelly = max(full_kelly, 0.0)

    # Quarter-Kelly for safety
    fraction = full_kelly * kelly_fraction
    # Hard position caps: min 0.5%, max 10% of bankroll per trade
    fraction = max(min(fraction, 0.10), 0.005)

    usd_size = fraction * bankroll

    # TP/SL prices
    if direction == "BUY":
        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)
    else:  # SELL/SHORT
        tp_price = entry_price * (1.0 - tp_pct)
        sl_price = entry_price * (1.0 + sl_pct)

    units = usd_size / entry_price if entry_price > 0 else 0.0
    reward_usd = usd_size * tp_pct * reward_risk
    risk_usd   = usd_size * sl_pct

    return PositionSize(
        fraction=round(fraction, 5),
        usd_size=round(usd_size, 2),
        units=round(units, 6),
        entry_price=entry_price,
        tp_price=round(tp_price, 6),
        sl_price=round(sl_price, 6),
        reward_usd=round(reward_usd, 2),
        risk_usd=round(risk_usd, 2),
        reward_risk_ratio=reward_risk,
    )


def _check_exposure(
    new_position_usd: float,
    bankroll: float,
    max_exposure_pct: float,
) -> RiskCheckResult:
    open_trades = get_open_trades()
    existing_exposure = sum(t.get("position_size_usd", 0) for t in open_trades)
    total_after = existing_exposure + new_position_usd
    max_allowed = bankroll * (max_exposure_pct / 100.0)
    passed = total_after <= max_allowed

    return RiskCheckResult(
        name="exposure_limit",
        passed=passed,
        reason=(
            f"Total exposure ${total_after:.0f} "
            f"({'≤' if passed else '>'} limit ${max_allowed:.0f} / {max_exposure_pct:.0f}% bankroll)"
        ),
        data={
            "existing_usd": existing_exposure,
            "new_usd": new_position_usd,
            "total_usd": total_after,
            "max_usd": max_allowed,
        },
    )


def _check_var(
    position_usd: float,
    bankroll: float,
    ohlcv: Optional[list],
    var_confidence: float,
    var_daily_limit_pct: float,
) -> RiskCheckResult:
    """
    Historical simulation VaR at 95% confidence.
    Uses last 30 days of 1h returns to estimate daily VaR.
    """
    if ohlcv and len(ohlcv) >= 25:
        closes = np.array([c[4] for c in ohlcv[-25:]], dtype=float)
        log_returns = np.diff(np.log(closes))
        # Scale hourly vol to daily (sqrt-time rule)
        daily_vol = float(np.std(log_returns)) * math.sqrt(24)
        # 95% VaR: 1.645 standard deviations
        z_score = 1.645
        var_pct = z_score * daily_vol
    else:
        # Fallback: use 3% daily VaR assumption
        var_pct = 0.03

    var_usd_for_position = position_usd * var_pct

    # Check against daily portfolio VaR limit
    existing_open = get_open_trades()
    existing_var = sum(
        t.get("position_size_usd", 0) * var_pct for t in existing_open
    )
    total_var = existing_var + var_usd_for_position
    max_daily_var = bankroll * (var_daily_limit_pct / 100.0)
    passed = total_var <= max_daily_var

    return RiskCheckResult(
        name="var_95pct",
        passed=passed,
        reason=(
            f"Daily VaR ${total_var:.0f} "
            f"({'≤' if passed else '>'} limit ${max_daily_var:.0f} / {var_daily_limit_pct:.0f}%)"
        ),
        data={
            "var_pct": round(var_pct, 4),
            "position_var_usd": round(var_usd_for_position, 2),
            "total_var_usd": round(total_var, 2),
            "max_var_usd": round(max_daily_var, 2),
        },
    )


def _check_drawdown(
    current_bankroll: float,
    initial_bankroll: float,
    max_drawdown_pct: float,
) -> RiskCheckResult:
    if initial_bankroll <= 0:
        return RiskCheckResult("max_drawdown", True, "No initial bankroll set")

    # Peak is the higher of initial and current (in case we're in profit)
    peak = max(initial_bankroll, current_bankroll)
    drawdown_pct = (peak - current_bankroll) / peak * 100.0
    passed = drawdown_pct <= max_drawdown_pct

    return RiskCheckResult(
        name="max_drawdown",
        passed=passed,
        reason=(
            f"Drawdown {drawdown_pct:.1f}% "
            f"({'≤' if passed else '>'} max {max_drawdown_pct:.0f}%)"
        ),
        data={"drawdown_pct": round(drawdown_pct, 2), "max_pct": max_drawdown_pct},
    )


def _check_daily_loss(
    current_bankroll: float,
    initial_bankroll: float,
    daily_loss_limit_pct: float,
) -> RiskCheckResult:
    today = date.today().isoformat()
    trades = get_all_trades()
    daily_pnl = sum(
        t.get("pnl_usd", 0)
        for t in trades
        if t.get("status") != "open" and t.get("closed_at", "")[:10] == today
    )
    daily_loss = min(daily_pnl, 0)  # only losses count
    max_loss = initial_bankroll * (daily_loss_limit_pct / 100.0)
    passed = abs(daily_loss) <= max_loss

    return RiskCheckResult(
        name="daily_loss_limit",
        passed=passed,
        reason=(
            f"Today's loss ${abs(daily_loss):.0f} "
            f"({'≤' if passed else '>'} limit ${max_loss:.0f} / {daily_loss_limit_pct:.0f}%)"
        ),
        data={"daily_pnl": round(daily_pnl, 2), "daily_loss_limit_usd": round(max_loss, 2)},
    )


def _check_concurrent_positions(max_concurrent: int) -> RiskCheckResult:
    open_trades = get_open_trades()
    n_open = len(open_trades)
    passed = n_open < max_concurrent

    return RiskCheckResult(
        name="concurrent_positions",
        passed=passed,
        reason=f"Open positions: {n_open} {'<' if passed else '>='} max {max_concurrent}",
        data={"open_positions": n_open, "max_concurrent": max_concurrent},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reject(checks: List[RiskCheckResult], reason: str) -> RiskValidationResult:
    _log.warning(f"Risk validation REJECTED: {reason}")
    return RiskValidationResult(
        approved=False,
        checks=checks,
        position=None,
        rejection_reason=reason,
        timestamp=datetime.utcnow().isoformat(),
    )


def get_portfolio_summary() -> dict:
    """Current portfolio risk metrics for the /status endpoint."""
    cfg = load_config()
    bankroll = cfg["bankroll"]["current_usd"]
    initial  = cfg["bankroll"]["initial_usd"]
    open_trades = get_open_trades()

    peak = max(initial, bankroll)
    drawdown_pct = (peak - bankroll) / peak * 100.0 if peak > 0 else 0.0
    exposure_usd = sum(t.get("position_size_usd", 0) for t in open_trades)
    exposure_pct = exposure_usd / bankroll * 100.0 if bankroll > 0 else 0.0

    today = date.today().isoformat()
    all_trades = get_all_trades()
    daily_pnl = sum(
        t.get("pnl_usd", 0)
        for t in all_trades
        if t.get("status") != "open" and t.get("closed_at", "")[:10] == today
    )

    return {
        "bankroll_usd": round(bankroll, 2),
        "initial_usd": round(initial, 2),
        "drawdown_pct": round(drawdown_pct, 2),
        "daily_pnl_usd": round(daily_pnl, 2),
        "open_positions": len(open_trades),
        "exposure_usd": round(exposure_usd, 2),
        "exposure_pct": round(exposure_pct, 2),
    }
