"""
compound.py — Step 5: Compound (Learn from every trade)
Runs after every trade closes:
  1. Classify failure reason (if loss)
  2. Write structured lesson to lessons.json
  3. Downrank market if repeated bad predictions
  4. Record prediction outcome for Brier score
  5. Trigger incremental model retraining
  6. Update performance metrics
"""
from datetime import datetime
from typing import Optional

from .config import load_config
from .logger import get_logger
from .storage import append_lesson, downrank_market, get_lessons
from .model import maybe_retrain

_log = get_logger("compound")


# ── Failure classification ────────────────────────────────────────────────────

def _classify_failure(trade: dict) -> str:
    """
    Classify why a losing trade failed.
    Returns one of: bad_prediction | bad_timing | bad_execution | external_shock
    """
    pnl_pct         = trade.get("pnl_pct", 0.0) or 0.0
    close_reason    = trade.get("close_reason", "")
    slippage_pct    = abs(trade.get("slippage_pct", 0.0) or 0.0)
    predicted_prob  = trade.get("predicted_probability", 0.5) or 0.5
    duration_s      = trade.get("duration_seconds", 0) or 0

    # Bad execution: high slippage or very fast SL hit (< 60s)
    if slippage_pct > 1.0 or (close_reason == "sl_hit" and duration_s < 60):
        return "bad_execution"

    # External shock: SL hit very quickly with no model signal issues
    if close_reason == "sl_hit" and duration_s < 300 and predicted_prob > 0.65:
        return "external_shock"

    # Bad timing: model was right direction but entered at wrong time
    if abs(pnl_pct) < 0.5 and close_reason == "sl_hit":
        return "bad_timing"

    # Default: bad prediction
    return "bad_prediction"


# ── Main post-mortem ──────────────────────────────────────────────────────────

def run_postmortem(trade: dict) -> dict:
    """
    Full Step 5 pipeline for one closed trade.
    Called by execution.close_trade() after every close.
    """
    trade_id = trade.get("id", "unknown")
    symbol   = trade.get("symbol", "unknown")
    outcome  = trade.get("outcome", "unknown")
    pnl_usd  = trade.get("pnl_usd", 0.0) or 0.0
    pnl_pct  = trade.get("pnl_pct", 0.0) or 0.0

    _log.info(
        f"Post-mortem: {trade_id} | {symbol} | {outcome.upper()} ${pnl_usd:+.2f}",
        data={"trade_id": trade_id, "symbol": symbol,
              "outcome": outcome, "pnl_usd": pnl_usd},
    )

    lesson = {
        "trade_id": trade_id,
        "symbol": symbol,
        "outcome": outcome,
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 4),
        "direction": trade.get("direction"),
        "entry_price": trade.get("entry_price"),
        "exit_price": trade.get("exit_price"),
        "predicted_probability": trade.get("predicted_probability"),
        "close_reason": trade.get("close_reason"),
        "duration_seconds": trade.get("duration_seconds"),
        "failure_type": None,
        "notes": "",
    }

    if outcome == "loss":
        failure_type = _classify_failure(trade)
        lesson["failure_type"] = failure_type

        notes = {
            "bad_prediction": "Model overestimated win probability. Review feature weights.",
            "bad_timing":     "Correct direction but poor entry. Consider tighter entry rules.",
            "bad_execution":  "Slippage or fast SL hit. Check spread and liquidity thresholds.",
            "external_shock": "Market shock — unpredictable event. Not a model failure.",
        }.get(failure_type, "Unknown failure type.")
        lesson["notes"] = notes

        _log.warning(
            f"Loss classified as '{failure_type}': {symbol} ${pnl_usd:.2f}",
            data={"failure_type": failure_type, "notes": notes},
        )

        # Downrank market after repeated bad predictions
        if failure_type == "bad_prediction":
            past_lessons = get_lessons()["lessons"]
            bad_preds = sum(
                1 for l in past_lessons
                if l.get("symbol") == symbol
                and l.get("failure_type") == "bad_prediction"
                and l.get("outcome") == "loss"
            )
            if bad_preds >= 2:
                downrank_market(
                    symbol,
                    reason=f"3+ bad predictions ({bad_preds + 1} total)",
                    duration_hours=24,
                )
                _log.warning(
                    f"Market downranked: {symbol} — {bad_preds + 1} bad predictions",
                    data={"symbol": symbol, "bad_predictions": bad_preds + 1},
                )
    else:
        lesson["notes"] = "Winning trade. Features and timing were well-aligned."
        _log.info(
            f"Win post-mortem: {symbol} ${pnl_usd:+.2f} in "
            f"{trade.get('duration_seconds', 0):.0f}s",
        )

    # Write lesson
    append_lesson(lesson)

    # Record outcome in Brier score tracker
    _record_prediction_outcome(trade)

    # Trigger incremental model retraining
    retrain_result = maybe_retrain()
    if retrain_result and retrain_result.get("status") == "trained":
        _log.info(
            f"Model retrained after trade close: "
            f"n={retrain_result.get('n_samples')} "
            f"Brier={retrain_result.get('brier_score', '?')}",
            data=retrain_result,
        )

    return lesson


def _record_prediction_outcome(trade: dict):
    """Feed the binary outcome back to the predictor's Brier tracker."""
    try:
        from .predictor import record_prediction_outcome
        outcome_binary = 1 if trade.get("outcome") == "win" else 0
        record_prediction_outcome(
            symbol=trade.get("symbol", ""),
            timestamp=trade.get("created_at", ""),
            outcome=outcome_binary,
        )
    except Exception as e:
        _log.warning(f"Could not record Brier outcome: {e}")


# ── Nightly consolidation ─────────────────────────────────────────────────────

def run_nightly_consolidation() -> dict:
    """
    Compute daily metrics and persist to performance_history.json.
    Called by APScheduler at midnight UTC.
    """
    from .storage import get_all_trades, append_daily_performance
    import statistics
    from datetime import date

    today = date.today().isoformat()
    all_trades = get_all_trades()
    today_trades = [
        t for t in all_trades
        if t.get("closed_at", "")[:10] == today
    ]

    if not today_trades:
        _log.info("Nightly consolidation: no trades today")
        return {"date": today, "n_trades": 0}

    pnl_list = [t.get("pnl_usd", 0) or 0 for t in today_trades]
    wins  = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p <= 0]

    win_rate   = len(wins) / len(pnl_list) if pnl_list else 0
    total_pnl  = sum(pnl_list)
    avg_win    = statistics.mean(wins)  if wins   else 0
    avg_loss   = statistics.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses else float("inf")

    # Sharpe (daily return vs std)
    cfg = load_config()
    bankroll = cfg["bankroll"]["current_usd"]
    daily_returns = [p / bankroll for p in pnl_list]
    sharpe_day = 0.0
    if len(daily_returns) > 1:
        std = statistics.stdev(daily_returns)
        mean = statistics.mean(daily_returns)
        sharpe_day = (mean / std) if std > 0 else 0.0

    record = {
        "n_trades": len(today_trades),
        "win_rate": round(win_rate, 4),
        "total_pnl_usd": round(total_pnl, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "sharpe_day": round(sharpe_day, 4),
        "bankroll_eod": round(bankroll, 2),
    }
    append_daily_performance(record)

    _log.info(
        f"Nightly consolidation: {len(today_trades)} trades | "
        f"win={win_rate:.0%} | PnL=${total_pnl:+.2f} | PF={profit_factor:.2f}",
        data=record,
    )
    return record
