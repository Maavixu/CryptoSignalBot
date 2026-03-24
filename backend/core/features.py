"""
features.py — Shared feature extractor.
CRITICAL: This same function is used for both training data generation
and live inference. Any divergence between training and inference features
is the #1 cause of silent model degradation. Never compute features twice.
"""
import math
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple
import numpy as np

# ── Feature vector definition ─────────────────────────────────────────────────
# Each feature has a name and index. Adding a new feature = append to end.
# Never reorder — it would invalidate saved models.

FEATURE_NAMES = [
    # Price momentum
    "price_change_1h_pct",       # 0
    "price_change_4h_pct",       # 1
    "price_change_24h_pct",      # 2
    "price_change_7d_pct",       # 3

    # Volume
    "volume_change_24h_pct",     # 4
    "volume_spike_ratio",        # 5  current / 7d avg

    # Technical indicators
    "rsi_14",                    # 6  [0, 100]
    "rsi_normalized",            # 7  (rsi - 50) / 50 → [-1, +1]
    "macd_signal",               # 8  macd - signal line, normalised
    "macd_histogram_slope",      # 9  change in histogram over last 3 bars
    "ma_50_distance_pct",        # 10 (price - MA50) / MA50 * 100
    "ma_200_distance_pct",       # 11 (price - MA200) / MA200 * 100
    "ma_50_200_cross",           # 12 1=golden cross, -1=death cross, 0=none
    "bb_position",               # 13 Bollinger position: 0=lower, 1=upper band
    "bb_width_pct",              # 14 band width as % of price

    # Volatility
    "atr_pct",                   # 15 ATR as % of price (normalised volatility)
    "realised_vol_24h",          # 16 std of 1h returns over 24h

    # Sentiment (from Research step)
    "sentiment_compound",        # 17 [-1, +1] cross-source weighted compound
    "sentiment_confidence",      # 18 [0, 1]
    "sentiment_agreement",       # 19 [0, 1] cross-source agreement
    "sentiment_price_gap",       # 20 sentiment - normalised price move

    # Market context
    "spread_pct",                # 21 bid-ask spread
    "depth_score",               # 22 log-normalised order book depth
    "anomaly_flag",              # 23 1 if any anomaly flagged by scanner
    "volume_rank_score",         # 24 [0, 1] rank among scanned markets

    # Target (not a feature — only used for training labels)
    # outcome: 1 = TP hit before SL, 0 = SL hit before TP
]

N_FEATURES = len(FEATURE_NAMES)


@dataclass
class FeatureVector:
    values: List[float]
    symbol: str
    timestamp: str = ""
    names: List[str] = field(default_factory=lambda: FEATURE_NAMES.copy())

    def to_numpy(self) -> np.ndarray:
        return np.array(self.values, dtype=np.float32).reshape(1, -1)

    def to_dict(self) -> dict:
        return dict(zip(self.names, self.values))

    def __len__(self):
        return len(self.values)


def extract_features(
    ohlcv: List[List],          # [[ts, open, high, low, close, volume], ...]
    symbol: str = "",
    sentiment_compound: float = 0.0,
    sentiment_confidence: float = 0.0,
    sentiment_agreement: float = 0.0,
    sentiment_price_gap: float = 0.0,
    spread_pct: float = 0.0,
    order_book_depth_usd: float = 0.0,
    volume_spike_ratio: float = 1.0,
    anomaly_flag: int = 0,
    volume_rank_score: float = 0.5,
) -> Optional[FeatureVector]:
    """
    Extract the full feature vector from OHLCV data + context.
    Returns None if insufficient data (< 200 candles).
    All features are normalised to comparable scales.
    """
    from datetime import datetime

    if not ohlcv or len(ohlcv) < 50:
        return None

    # Extract arrays
    closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
    highs  = np.array([c[2] for c in ohlcv], dtype=np.float64)
    lows   = np.array([c[3] for c in ohlcv], dtype=np.float64)
    vols   = np.array([c[5] for c in ohlcv], dtype=np.float64)

    n = len(closes)
    current_price = closes[-1]

    # ── Price momentum ────────────────────────────────────────────────────────
    def pct_change(lookback: int) -> float:
        if n <= lookback:
            return 0.0
        base = closes[-(lookback + 1)]
        return ((current_price - base) / base * 100) if base != 0 else 0.0

    price_change_1h  = pct_change(1)
    price_change_4h  = pct_change(4)
    price_change_24h = pct_change(24)
    price_change_7d  = pct_change(min(168, n - 1))

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_24h = float(np.sum(vols[-24:])) if n >= 24 else float(np.sum(vols))
    vol_prev_24h = float(np.sum(vols[-48:-24])) if n >= 48 else vol_24h
    volume_change_24h = ((vol_24h - vol_prev_24h) / vol_prev_24h * 100) \
        if vol_prev_24h > 0 else 0.0

    # ── RSI (14 period) ───────────────────────────────────────────────────────
    rsi_val = _compute_rsi(closes, period=14)
    rsi_normalized = (rsi_val - 50.0) / 50.0   # [-1, +1]

    # ── MACD (12, 26, 9) ─────────────────────────────────────────────────────
    macd_line, signal_line, histogram = _compute_macd(closes)
    # Normalise by current price
    macd_signal_norm = (macd_line - signal_line) / current_price * 100 \
        if current_price > 0 else 0.0
    histogram_slope = (histogram[-1] - histogram[-4]) / abs(histogram[-4]) \
        if len(histogram) >= 4 and histogram[-4] != 0 else 0.0
    histogram_slope = float(np.clip(histogram_slope, -5.0, 5.0))

    # ── Moving averages ───────────────────────────────────────────────────────
    ma50 = float(np.mean(closes[-50:])) if n >= 50 else current_price
    ma200 = float(np.mean(closes[-200:])) if n >= 200 else float(np.mean(closes))

    ma50_dist  = (current_price - ma50)  / ma50  * 100 if ma50  > 0 else 0.0
    ma200_dist = (current_price - ma200) / ma200 * 100 if ma200 > 0 else 0.0

    # Golden/death cross signal
    if n >= 201:
        prev_ma50  = float(np.mean(closes[-51:-1]))
        prev_ma200 = float(np.mean(closes[-201:-1]))
        if prev_ma50 < prev_ma200 and ma50 >= ma200:
            ma_cross = 1.0    # golden cross
        elif prev_ma50 > prev_ma200 and ma50 <= ma200:
            ma_cross = -1.0   # death cross
        else:
            ma_cross = 0.0
    else:
        ma_cross = 0.0

    # ── Bollinger Bands (20 period, 2σ) ──────────────────────────────────────
    bb_period = min(20, n)
    bb_closes = closes[-bb_period:]
    bb_mean = float(np.mean(bb_closes))
    bb_std  = float(np.std(bb_closes))
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_position = (current_price - bb_lower) / bb_width if bb_width > 0 else 0.5
    bb_position = float(np.clip(bb_position, 0.0, 1.0))
    bb_width_pct = (bb_width / bb_mean * 100) if bb_mean > 0 else 0.0

    # ── ATR (Average True Range, 14 period) ───────────────────────────────────
    atr = _compute_atr(highs, lows, closes, period=14)
    atr_pct = (atr / current_price * 100) if current_price > 0 else 0.0

    # ── Realised volatility (std of 1h log returns over 24h) ─────────────────
    if n >= 25:
        log_returns = np.diff(np.log(closes[-25:]))
        realised_vol = float(np.std(log_returns) * 100)
    else:
        realised_vol = 0.0

    # ── Market context ────────────────────────────────────────────────────────
    depth_score = min(math.log10(max(order_book_depth_usd, 1)) / 7.0, 1.0)

    # ── Assemble vector ───────────────────────────────────────────────────────
    values = [
        float(np.clip(price_change_1h,  -50, 50)),
        float(np.clip(price_change_4h,  -50, 50)),
        float(np.clip(price_change_24h, -50, 50)),
        float(np.clip(price_change_7d,  -100, 100)),
        float(np.clip(volume_change_24h, -500, 500)),
        float(np.clip(volume_spike_ratio, 0, 20)),
        float(np.clip(rsi_val, 0, 100)),
        float(np.clip(rsi_normalized, -1, 1)),
        float(np.clip(macd_signal_norm, -5, 5)),
        float(np.clip(histogram_slope, -5, 5)),
        float(np.clip(ma50_dist,  -50, 50)),
        float(np.clip(ma200_dist, -100, 100)),
        float(np.clip(ma_cross, -1, 1)),
        float(np.clip(bb_position, 0, 1)),
        float(np.clip(bb_width_pct, 0, 20)),
        float(np.clip(atr_pct, 0, 20)),
        float(np.clip(realised_vol, 0, 20)),
        float(np.clip(sentiment_compound, -1, 1)),
        float(np.clip(sentiment_confidence, 0, 1)),
        float(np.clip(sentiment_agreement, 0, 1)),
        float(np.clip(sentiment_price_gap, -2, 2)),
        float(np.clip(spread_pct, 0, 5)),
        float(np.clip(depth_score, 0, 1)),
        float(anomaly_flag),
        float(np.clip(volume_rank_score, 0, 1)),
    ]

    assert len(values) == N_FEATURES, f"Feature count mismatch: {len(values)} != {N_FEATURES}"

    return FeatureVector(
        values=values,
        symbol=symbol,
        timestamp=datetime.utcnow().isoformat(),
    )


# ── Technical indicator helpers ───────────────────────────────────────────────

def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 10):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[float, float, np.ndarray]:
    """Returns (macd_line, signal_line, histogram_array)."""
    if len(closes) < slow + signal:
        return 0.0, 0.0, np.zeros(signal)

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_series = ema_fast - ema_slow
    signal_series = _ema(macd_series, signal)
    histogram = macd_series[-signal:] - signal_series[-signal:]
    return float(macd_series[-1]), float(signal_series[-1]), histogram


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    k = 2.0 / (period + 1)
    ema = np.zeros(len(arr))
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = arr[i] * k + ema[i - 1] * (1 - k)
    return ema


def _compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    if len(closes) < period + 1:
        return float(np.mean(highs - lows))
    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        true_ranges.append(tr)
    return float(np.mean(true_ranges[-period:]))
