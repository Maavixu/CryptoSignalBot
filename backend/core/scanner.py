"""
scanner.py — Step 1: Scan
Fetches markets, applies filters, detects anomalies, and ranks
by opportunity score. Output feeds directly into Step 2: Research.
"""
import asyncio
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..core.config import load_config
from ..core.exchange import get_adapter
from ..core.logger import get_logger
from ..core.storage import get_downranked_markets

_log = get_logger("scanner")

STEP = "scan"


@dataclass
class MarketSnapshot:
    symbol: str
    last_price: float
    volume_24h_usd: float
    price_change_pct_24h: float
    price_change_pct_1h: float
    spread_pct: float
    order_book_depth_usd: float
    volume_7d_avg_usd: float
    volume_spike_ratio: float        # current 24h vs 7d avg
    opportunity_score: float
    anomaly_flags: List[str] = field(default_factory=list)
    is_tradable: bool = True
    scanned_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class ScanResult:
    timestamp: str
    total_markets_fetched: int
    markets_passed_filter: int
    markets_flagged_anomaly: int
    top_opportunities: List[MarketSnapshot]
    all_tradable: List[MarketSnapshot]
    duration_seconds: float
    exchange: str


# ── Core scanner logic ────────────────────────────────────────────────────────

async def run_scan() -> ScanResult:
    """
    Full Step 1 pipeline:
    1. Fetch all tickers
    2. Filter by volume / spread / depth
    3. Detect anomalies
    4. Score and rank
    5. Apply lesson-based downranks
    Returns ScanResult with ranked opportunities.
    """
    start_ts = time.time()
    cfg = load_config()
    scan_cfg = cfg["scanner"]
    adapter = get_adapter()

    _log.info("Scan started", data={"config": scan_cfg})

    # ── 1. Fetch tickers ───────────────────────────────────────────────────
    tickers = await adapter.fetch_tickers()
    total_fetched = len(tickers)
    _log.info(f"Fetched {total_fetched} tickers")

    # Filter to USDT quote pairs only
    usdt_pairs = {k: v for k, v in tickers.items() if k.endswith("/USDT")}
    _log.info(f"{len(usdt_pairs)} USDT pairs available")

    # ── 2. Apply basic filters ─────────────────────────────────────────────
    min_vol = scan_cfg["min_volume_usd_24h"]
    max_spread = scan_cfg["max_spread_pct"]
    min_depth = scan_cfg["min_order_book_depth_usd"]

    snapshots: List[MarketSnapshot] = []
    order_book_tasks = []

    # Pre-filter by volume before fetching order books (saves API calls)
    pre_filtered = [
        (sym, t) for sym, t in usdt_pairs.items()
        if (t.get("quoteVolume") or 0) >= min_vol
    ]
    _log.info(
        f"{len(pre_filtered)} pairs pass volume filter (>{min_vol/1e6:.1f}M USD)"
    )

    # Cap at top_n_pairs to avoid too many order book calls
    top_n = scan_cfg["top_n_pairs"]
    pre_filtered.sort(key=lambda x: x[1].get("quoteVolume", 0), reverse=True)
    pre_filtered = pre_filtered[:top_n]

    # Fetch order books concurrently (chunked to respect rate limits)
    async def fetch_ob_safe(sym: str) -> Tuple[str, dict]:
        try:
            ob = await adapter.fetch_order_book(sym, limit=20)
            return sym, ob
        except Exception as e:
            _log.debug(f"Order book fetch failed for {sym}: {e}")
            return sym, {}

    chunk_size = 10
    order_books: Dict[str, dict] = {}
    for i in range(0, len(pre_filtered), chunk_size):
        chunk = pre_filtered[i:i + chunk_size]
        results = await asyncio.gather(*[fetch_ob_safe(sym) for sym, _ in chunk])
        order_books.update(dict(results))
        await asyncio.sleep(0.1)

    # ── 3. Build snapshots with spread + depth ─────────────────────────────
    downranked = get_downranked_markets()

    for sym, ticker in pre_filtered:
        ob = order_books.get(sym, {})
        spread_pct = adapter.get_spread_pct(ob) if ob else 999.0
        depth_usd = adapter.get_order_book_depth_usd(ob) if ob else 0.0

        if spread_pct > max_spread:
            continue
        if depth_usd < min_depth:
            continue

        # ── 4. Anomaly detection ───────────────────────────────────────────
        anomalies = []
        price_change_24h = ticker.get("percentage", 0.0) or 0.0

        # 1h price change (approximate from high/low if no dedicated field)
        price_change_1h = _estimate_1h_change(ticker)

        if abs(price_change_1h) >= scan_cfg["anomaly_price_change_pct_1h"]:
            direction = "up" if price_change_1h > 0 else "down"
            anomalies.append(f"price_spike_1h_{direction}:{price_change_1h:.1f}%")

        # Volume spike vs 7-day average (estimated from 24h vol)
        vol_24h = ticker.get("quoteVolume", 0.0) or 0.0
        vol_7d_avg = _estimate_7d_avg_volume(sym, vol_24h)
        spike_ratio = vol_24h / vol_7d_avg if vol_7d_avg > 0 else 1.0

        if spike_ratio >= scan_cfg["anomaly_volume_multiplier_7d"]:
            anomalies.append(f"volume_spike:{spike_ratio:.1f}x")

        # ── 5. Opportunity score ───────────────────────────────────────────
        score = _compute_opportunity_score(
            vol_24h=vol_24h,
            spread_pct=spread_pct,
            price_change_pct_24h=price_change_24h,
            spike_ratio=spike_ratio,
            depth_usd=depth_usd,
        )

        # Apply lesson-based penalty
        if sym in downranked:
            score *= 0.3
            anomalies.append(f"lesson_downrank:{downranked[sym]['reason']}")

        snap = MarketSnapshot(
            symbol=sym,
            last_price=ticker.get("last", 0.0),
            volume_24h_usd=vol_24h,
            price_change_pct_24h=price_change_24h,
            price_change_pct_1h=price_change_1h,
            spread_pct=spread_pct,
            order_book_depth_usd=depth_usd,
            volume_7d_avg_usd=vol_7d_avg,
            volume_spike_ratio=spike_ratio,
            opportunity_score=score,
            anomaly_flags=anomalies,
            is_tradable=True,
            scanned_at=datetime.utcnow().isoformat(),
        )
        snapshots.append(snap)

    # ── 6. Sort and select top opportunities ──────────────────────────────
    snapshots.sort(key=lambda s: s.opportunity_score, reverse=True)
    top_n_opps = scan_cfg["opportunity_score_top_n"]
    top_opportunities = snapshots[:top_n_opps]
    flagged = [s for s in snapshots if s.anomaly_flags]

    duration = time.time() - start_ts
    result = ScanResult(
        timestamp=datetime.utcnow().isoformat(),
        total_markets_fetched=total_fetched,
        markets_passed_filter=len(snapshots),
        markets_flagged_anomaly=len(flagged),
        top_opportunities=top_opportunities,
        all_tradable=snapshots,
        duration_seconds=round(duration, 2),
        exchange=adapter.exchange_id,
    )

    _log.info(
        f"Scan complete: {len(snapshots)} tradable markets, "
        f"{len(top_opportunities)} top opportunities, "
        f"{len(flagged)} anomalies detected",
        data={
            "duration_s": round(duration, 2),
            "top_pairs": [s.symbol for s in top_opportunities[:5]],
            "anomalies": [
                {"symbol": s.symbol, "flags": s.anomaly_flags}
                for s in flagged[:5]
            ],
        },
    )

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_1h_change(ticker: dict) -> float:
    """
    Estimate 1h price change. Exchanges often don't provide this directly.
    We use a heuristic: (last - open) / open scaled to 1h window.
    In production, fetch 1h OHLCV for accuracy.
    """
    import random
    # Mock: derive from 24h change with some noise
    change_24h = ticker.get("percentage", 0.0) or 0.0
    noise = random.gauss(0, abs(change_24h) * 0.3 + 0.5)
    return change_24h / 4 + noise  # rough 1h approximation


def _estimate_7d_avg_volume(symbol: str, vol_24h: float) -> float:
    """
    Estimate 7-day average daily volume.
    In production: fetch 7 days of daily candles and average.
    Here we use a heuristic with slight randomisation for demo.
    """
    import random
    # Simulate: 7d avg is typically 60-140% of today's volume
    factor = random.uniform(0.6, 1.4)
    return vol_24h * factor


def _compute_opportunity_score(
    vol_24h: float,
    spread_pct: float,
    price_change_pct_24h: float,
    spike_ratio: float,
    depth_usd: float,
) -> float:
    """
    Composite opportunity score in [0, 100].
    Higher = more worth researching.

    Components:
    - Volume score: log-normalised, higher is better
    - Spread score: lower spread = higher score
    - Momentum score: absolute price movement
    - Spike bonus: volume anomalies attract attention
    - Depth score: more depth = safer execution
    """
    import math

    # Volume (0-40 pts): log scale, cap at 1B
    vol_capped = min(vol_24h, 1e9)
    vol_score = (math.log10(max(vol_capped, 1)) / math.log10(1e9)) * 40

    # Spread (0-20 pts): perfect spread=0 gives 20, spread>=0.5% gives 0
    spread_score = max(0.0, (0.5 - spread_pct) / 0.5) * 20

    # Momentum (0-20 pts): abs price change, cap at 15%
    momentum_score = min(abs(price_change_pct_24h) / 15, 1.0) * 20

    # Spike bonus (0-10 pts): volume spike up to 5x
    spike_score = min((spike_ratio - 1) / 4, 1.0) * 10 if spike_ratio > 1 else 0

    # Depth (0-10 pts): depth up to 5M USD
    depth_score = min(depth_usd / 5e6, 1.0) * 10

    total = vol_score + spread_score + momentum_score + spike_score + depth_score
    return round(total, 2)


# ── Scheduler-facing entry point ──────────────────────────────────────────────

def run_scan_sync() -> Optional[ScanResult]:
    """Synchronous wrapper for APScheduler."""
    try:
        return asyncio.run(run_scan())
    except Exception as e:
        _log.error(f"Scan failed: {e}")
        return None
