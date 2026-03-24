"""
predictor.py — Step 3: Predict
Orchestrates the full prediction pipeline:
  1. Fetch OHLCV for the symbol
  2. Extract feature vector (shared with training)
  3. Run ensemble (XGBoost + MLP)
  4. Compute market baseline probability from volatility
  5. Calculate edge = p_model - p_market
  6. Apply direction logic (BUY/SELL/SKIP)
  7. Log every prediction for Brier score tracking
  8. Trigger model retraining if dataset grew enough
"""
import asyncio
import math
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

from .config import load_config
from .exchange import get_adapter
from .features import extract_features, FeatureVector
from .model import get_model, PredictionResult, maybe_retrain
from .logger import get_logger
from .researcher import ResearchBrief

_log = get_logger("predict")

# In-memory prediction log for Brier score tracking
# {prediction_id: {p_win, symbol, timestamp, outcome (filled on trade close)}}
_prediction_log: Dict[str, dict] = {}


async def run_prediction(
    symbol: str,
    market_snapshot,          # MarketSnapshot from scanner
    research_brief: Optional[ResearchBrief] = None,
) -> Optional[PredictionResult]:
    """
    Full Step 3 prediction pipeline for one symbol.
    Returns PredictionResult or None if data insufficient.
    """
    start_ts = time.time()
    cfg = load_config()
    adapter = get_adapter()

    _log.info(f"Prediction started: {symbol}", data={"symbol": symbol})

    # ── 1. Fetch OHLCV (1h candles, 200 periods) ──────────────────────────
    ohlcv = await adapter.fetch_ohlcv(symbol, timeframe="1h", limit=210)
    if not ohlcv or len(ohlcv) < 50:
        _log.warning(f"Insufficient OHLCV data for {symbol}: {len(ohlcv) if ohlcv else 0} candles")
        return None

    # ── 2. Extract feature vector ──────────────────────────────────────────
    sentiment_compound    = research_brief.adjusted_compound     if research_brief else 0.0
    sentiment_confidence  = research_brief.consensus_confidence  if research_brief else 0.0
    sentiment_agreement   = research_brief.source_agreement      if research_brief else 0.0
    sentiment_price_gap   = research_brief.sentiment_price_gap   if research_brief else 0.0
    anomaly_flag          = 1 if market_snapshot.anomaly_flags else 0

    fv = extract_features(
        ohlcv=ohlcv,
        symbol=symbol,
        sentiment_compound=sentiment_compound,
        sentiment_confidence=sentiment_confidence,
        sentiment_agreement=sentiment_agreement,
        sentiment_price_gap=sentiment_price_gap,
        spread_pct=market_snapshot.spread_pct,
        order_book_depth_usd=market_snapshot.order_book_depth_usd,
        volume_spike_ratio=market_snapshot.volume_spike_ratio,
        anomaly_flag=anomaly_flag,
        volume_rank_score=market_snapshot.opportunity_score / 100.0,
    )

    if fv is None:
        _log.warning(f"Feature extraction failed for {symbol}")
        return None

    # ── 3. Ensemble prediction ────────────────────────────────────────────
    model = get_model()
    ensemble_prob, xgb_prob, mlp_prob = model.predict(fv.to_numpy())

    _log.info(
        f"Ensemble prediction for {symbol}: "
        f"ensemble={ensemble_prob:.3f} xgb={xgb_prob:.3f} mlp={mlp_prob:.3f}",
        data={
            "symbol": symbol,
            "ensemble": round(ensemble_prob, 4),
            "xgb": round(xgb_prob, 4),
            "mlp": round(mlp_prob, 4),
            "model_trained": model.is_trained,
            "n_training_samples": model.n_samples,
        },
    )

    # ── 4. Market baseline probability ────────────────────────────────────
    # Derived from recent volatility: in a fair coin-flip market driven
    # purely by volatility, p_market ≈ 0.50 ± volatility adjustment.
    # We use a simple approximation based on ATR and TP/SL ratio.
    p_market = _compute_market_baseline(ohlcv, cfg)

    # ── 5. Edge calculation ────────────────────────────────────────────────
    edge = ensemble_prob - p_market
    edge_pct = edge * 100.0
    min_edge = cfg["prediction"]["min_edge_pct"]

    # ── 6. Direction ───────────────────────────────────────────────────────
    direction = _determine_direction(
        ensemble_prob=ensemble_prob,
        sentiment_compound=sentiment_compound,
        price_change_24h=market_snapshot.price_change_pct_24h,
        gap_signal=research_brief.gap_signal if research_brief else "neutral",
    )

    # ── 7. Confidence band ─────────────────────────────────────────────────
    model_disagreement = abs(xgb_prob - mlp_prob)
    if edge_pct >= 8 and model_disagreement < 0.10:
        confidence = "high"
    elif edge_pct >= min_edge and model_disagreement < 0.20:
        confidence = "medium"
    else:
        confidence = "low"

    # ── 8. Feature importances (top 5) ────────────────────────────────────
    importances = model.get_feature_importance()
    top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5])

    result = PredictionResult(
        symbol=symbol,
        p_win=round(ensemble_prob, 4),
        p_market=round(p_market, 4),
        edge=round(edge, 4),
        edge_pct=round(edge_pct, 2),
        direction=direction if edge_pct >= min_edge else "SKIP",
        xgb_prob=round(xgb_prob, 4),
        mlp_prob=round(mlp_prob, 4),
        confidence=confidence,
        feature_importances=top_features,
        timestamp=datetime.utcnow().isoformat(),
    )

    # ── 9. Log prediction for Brier score tracking ────────────────────────
    pred_id = str(uuid.uuid4())[:12]
    _prediction_log[pred_id] = {
        "id": pred_id,
        "symbol": symbol,
        "p_win": ensemble_prob,
        "direction": result.direction,
        "timestamp": result.timestamp,
        "outcome": None,  # filled when trade closes
    }

    duration = round(time.time() - start_ts, 2)

    _log.info(
        f"Predict complete: {symbol} → {result.direction} "
        f"| p_win={ensemble_prob:.3f} p_mkt={p_market:.3f} "
        f"edge={edge_pct:+.1f}% [{confidence}]",
        data={
            "symbol": symbol,
            "direction": result.direction,
            "p_win": round(ensemble_prob, 4),
            "p_market": round(p_market, 4),
            "edge_pct": round(edge_pct, 2),
            "confidence": confidence,
            "has_edge": result.has_edge,
            "model_disagreement": round(model_disagreement, 4),
            "duration_s": duration,
        },
    )

    # ── 10. Trigger incremental retraining if due ─────────────────────────
    retrain_result = maybe_retrain()
    if retrain_result:
        _log.info(
            f"Auto-retrain: {retrain_result.get('status')} "
            f"({retrain_result.get('n_samples', 0)} samples)",
            data=retrain_result,
        )

    return result


def record_prediction_outcome(symbol: str, timestamp: str, outcome: int) -> Optional[float]:
    """
    Record the binary outcome for a past prediction.
    Called when a trade closes (outcome: 1=win/TP, 0=loss/SL).
    Returns updated Brier contribution for this prediction.
    """
    # Find the matching prediction by symbol + approximate timestamp
    for pred_id, pred in _prediction_log.items():
        if pred["symbol"] == symbol and pred["outcome"] is None:
            if abs(_ts_diff_seconds(pred["timestamp"], timestamp)) < 300:
                pred["outcome"] = outcome
                # Brier contribution: (p - outcome)^2
                brier_contrib = (pred["p_win"] - outcome) ** 2
                _log.info(
                    f"Prediction outcome recorded: {symbol} outcome={outcome} "
                    f"p_win={pred['p_win']:.3f} Brier_contrib={brier_contrib:.4f}",
                    data={"symbol": symbol, "outcome": outcome,
                          "p_win": pred["p_win"], "brier": brier_contrib},
                )
                return brier_contrib
    return None


def compute_rolling_brier_score() -> Optional[float]:
    """Compute rolling Brier score over all resolved predictions."""
    resolved = [p for p in _prediction_log.values() if p["outcome"] is not None]
    if not resolved:
        return None
    bs = sum((p["p_win"] - p["outcome"]) ** 2 for p in resolved) / len(resolved)
    return round(bs, 4)


def get_prediction_log() -> List[dict]:
    return list(_prediction_log.values())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_market_baseline(ohlcv: List[List], cfg: dict) -> float:
    """
    Baseline probability from the market's perspective.
    Based on the Kelly-implied probability for the TP/SL ratio:
    For a 2:1 reward:risk ratio, break-even p = 1/(1+b) = 1/3 ≈ 0.333
    Adjusted upward by realized volatility (higher vol = more uncertainty → closer to 0.5)
    """
    import numpy as np
    b = cfg["trade"]["default_reward_risk_ratio"]
    break_even_p = 1.0 / (1.0 + b)  # e.g., 0.333 for 2:1

    # Volatility adjustment — high vol → p_market closer to 0.5
    if len(ohlcv) >= 25:
        closes = np.array([c[4] for c in ohlcv[-25:]])
        log_returns = np.diff(np.log(closes))
        vol = float(np.std(log_returns))
        # vol in range [0, 0.05] maps adjustment [0, 0.05]
        vol_adj = min(vol * 2, 0.10)
    else:
        vol_adj = 0.05

    # Blend break-even p with 0.5 based on vol
    p_market = break_even_p + (0.5 - break_even_p) * (vol_adj / 0.10)
    return round(float(np.clip(p_market, 0.30, 0.50)), 4)


def _determine_direction(
    ensemble_prob: float,
    sentiment_compound: float,
    price_change_24h: float,
    gap_signal: str,
) -> str:
    """
    Map ensemble probability and context to BUY / SELL / SKIP.
    High probability = BUY direction.
    If sentiment is bearish and price recently surged = SELL.
    """
    # Strong model confidence → BUY
    if ensemble_prob >= 0.60:
        # Cross-check: don't buy into overbought + bearish sentiment reversal
        if sentiment_compound < -0.2 and price_change_24h > 10 and gap_signal == "bearish_gap":
            return "SELL"
        return "BUY"
    elif ensemble_prob <= 0.40:
        # Model says high chance of SL hit → potential SELL/SHORT signal
        if sentiment_compound < -0.1:
            return "SELL"
        return "SKIP"
    else:
        # 0.40–0.60 range: too uncertain unless gap signal is strong
        if gap_signal == "bullish_gap" and sentiment_compound > 0.2:
            return "BUY"
        elif gap_signal == "bearish_gap" and sentiment_compound < -0.2:
            return "SELL"
        return "SKIP"


def _ts_diff_seconds(ts1: str, ts2: str) -> float:
    """Difference in seconds between two ISO timestamps."""
    from datetime import datetime
    try:
        fmt = "%Y-%m-%dT%H:%M:%S"
        t1 = datetime.fromisoformat(ts1[:19])
        t2 = datetime.fromisoformat(ts2[:19])
        return (t1 - t2).total_seconds()
    except Exception:
        return 9999.0


# ── Sync wrapper ──────────────────────────────────────────────────────────────

def run_prediction_sync(
    symbol: str,
    market_snapshot,
    research_brief=None,
) -> Optional[PredictionResult]:
    try:
        return asyncio.run(run_prediction(symbol, market_snapshot, research_brief))
    except Exception as e:
        _log.error(f"Prediction sync failed for {symbol}: {e}")
        return None
