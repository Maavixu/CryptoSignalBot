"""
researcher.py — Step 2: Research
Orchestrates parallel research agents for each market flagged by the scanner.
Produces a ResearchBrief per symbol that flows into Step 3: Predict.

Pipeline per symbol:
  1. Run RSS, Reddit, Twitter agents concurrently
  2. Sanitise all content
  3. Run NLP sentiment on each source
  4. Aggregate across sources with source weighting
  5. Compute sentiment-price gap
  6. Cross-reference sources for consensus vs divergence
  7. Apply lesson penalties from lessons.json
  8. Return ResearchBrief
"""
import asyncio
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .config import load_config
from .logger import get_logger
from .sentiment import get_analyser, AggregatedSentiment, SentimentScore
from .sources.rss import fetch_rss
from .sources.reddit import fetch_reddit
from .sources.twitter import fetch_twitter
from .storage import get_lessons

_log = get_logger("research")

# Source reliability weights — RSS/news weighted highest as most factual
SOURCE_WEIGHTS = {
    "rss": 1.0,
    "reddit": 0.75,
    "twitter": 0.6,
}


@dataclass
class SourceBrief:
    source: str
    label: str          # bullish / bearish / neutral
    compound: float     # [-1, +1]
    confidence: float
    item_count: int
    weight: float       # source reliability weight
    top_snippets: List[str] = field(default_factory=list)


@dataclass
class ResearchBrief:
    symbol: str
    timestamp: str

    # Aggregated cross-source sentiment
    consensus_label: str        # "bullish" | "bearish" | "neutral"
    consensus_compound: float   # weighted avg across all sources
    consensus_confidence: float

    # Per-source breakdown
    sources: List[SourceBrief] = field(default_factory=list)

    # Sentiment-price gap analysis
    price_change_24h_pct: float = 0.0
    sentiment_price_gap: float = 0.0   # positive = sentiment more bullish than price move
    gap_signal: str = "neutral"         # "bullish_gap" | "bearish_gap" | "aligned" | "neutral"

    # Cross-source consensus quality
    source_agreement: float = 0.0      # [0,1] — how much sources agree
    divergent_sources: List[str] = field(default_factory=list)

    # Lesson-adjusted score (penalised if market has bad history)
    lesson_penalty: float = 0.0
    adjusted_compound: float = 0.0

    # Metadata
    total_items_analysed: int = 0
    duration_seconds: float = 0.0
    research_quality: str = "low"   # "low" | "medium" | "high"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Main research pipeline ────────────────────────────────────────────────────

async def run_research(
    symbol: str,
    price_change_24h_pct: float = 0.0,
    anomaly_flags: List[str] = None,
) -> ResearchBrief:
    """
    Full Step 2 pipeline for a single symbol.
    Runs all source agents concurrently.
    """
    start_ts = time.time()
    cfg = load_config()
    res_cfg = cfg["research"]
    analyser = get_analyser()

    _log.info(
        f"Research started: {symbol}",
        data={"symbol": symbol, "price_change_24h": price_change_24h_pct},
    )

    # ── 1. Concurrent source fetching ─────────────────────────────────────
    rss_task = fetch_rss(
        res_cfg["rss_feeds"], symbol,
        max_per_source=res_cfg["max_articles_per_source"],
    )
    reddit_task = fetch_reddit(
        res_cfg["reddit_subreddits"], symbol,
        max_posts=res_cfg["max_articles_per_source"],
    )
    twitter_task = fetch_twitter(
        symbol,
        max_tweets=res_cfg["max_articles_per_source"],
    )

    rss_results, reddit_results, twitter_results = await asyncio.gather(
        rss_task, reddit_task, twitter_task, return_exceptions=True,
    )

    # Handle exceptions from individual sources gracefully
    rss_items = rss_results if isinstance(rss_results, list) else []
    reddit_items = reddit_results if isinstance(reddit_results, list) else []
    twitter_items = twitter_results if isinstance(twitter_results, list) else []

    if isinstance(rss_results, Exception):
        _log.error(f"RSS agent failed for {symbol}: {rss_results}")
    if isinstance(reddit_results, Exception):
        _log.error(f"Reddit agent failed for {symbol}: {reddit_results}")
    if isinstance(twitter_results, Exception):
        _log.error(f"Twitter agent failed for {symbol}: {twitter_results}")

    # ── 2. Sentiment scoring per source ───────────────────────────────────
    min_conf = res_cfg["sentiment_min_confidence"]

    source_briefs: List[SourceBrief] = []

    # RSS
    if rss_items:
        texts = [a.combined_text for a in rss_items]
        scores = analyser.score_batch(texts)
        agg = analyser.aggregate(scores, "rss", symbol, min_confidence=min_conf)
        source_briefs.append(_make_source_brief("rss", agg, scores))
        _log.info(
            f"RSS sentiment for {symbol}: {agg.label} "
            f"(compound={agg.compound_avg:+.3f}, n={agg.total_items})",
            data={"source": "rss", "symbol": symbol, "label": agg.label,
                  "compound": agg.compound_avg, "items": agg.total_items},
        )

    # Reddit
    if reddit_items:
        # Weight by post score (upvotes signal community consensus)
        texts = [p.combined_text for p in reddit_items]
        scores = analyser.score_batch(texts)
        # Boost confidence for high-scoring posts
        for i, post in enumerate(reddit_items):
            if i < len(scores) and post.score > 1000:
                scores[i] = SentimentScore(
                    label=scores[i].label,
                    compound=scores[i].compound,
                    confidence=min(scores[i].confidence * 1.3, 1.0),
                    positive=scores[i].positive,
                    negative=scores[i].negative,
                    neutral=scores[i].neutral,
                    text_snippet=scores[i].text_snippet,
                )
        agg = analyser.aggregate(scores, "reddit", symbol, min_confidence=min_conf)
        source_briefs.append(_make_source_brief("reddit", agg, scores))
        _log.info(
            f"Reddit sentiment for {symbol}: {agg.label} "
            f"(compound={agg.compound_avg:+.3f}, n={agg.total_items})",
            data={"source": "reddit", "symbol": symbol, "label": agg.label,
                  "compound": agg.compound_avg, "items": agg.total_items},
        )

    # Twitter
    if twitter_items:
        texts = [t.combined_text for t in twitter_items]
        scores = analyser.score_batch(texts)
        # Boost confidence for high-engagement tweets
        for i, tweet in enumerate(twitter_items):
            if i < len(scores) and tweet.weight > 2.0:
                scores[i] = SentimentScore(
                    label=scores[i].label,
                    compound=scores[i].compound,
                    confidence=min(scores[i].confidence * tweet.weight * 0.5, 1.0),
                    positive=scores[i].positive,
                    negative=scores[i].negative,
                    neutral=scores[i].neutral,
                    text_snippet=scores[i].text_snippet,
                )
        agg = analyser.aggregate(scores, "twitter", symbol, min_confidence=min_conf)
        source_briefs.append(_make_source_brief("twitter", agg, scores))
        _log.info(
            f"Twitter sentiment for {symbol}: {agg.label} "
            f"(compound={agg.compound_avg:+.3f}, n={agg.total_items})",
            data={"source": "twitter", "symbol": symbol, "label": agg.label,
                  "compound": agg.compound_avg, "items": agg.total_items},
        )

    # ── 3. Cross-source aggregation ───────────────────────────────────────
    if not source_briefs:
        _log.warning(f"No sources returned data for {symbol}, returning neutral brief")
        return _neutral_brief(symbol, price_change_24h_pct, time.time() - start_ts)

    consensus = _aggregate_cross_source(source_briefs)
    consensus_label = consensus["label"]
    consensus_compound = consensus["compound"]
    consensus_confidence = consensus["confidence"]
    agreement = consensus["agreement"]
    divergent = consensus["divergent"]

    # ── 4. Sentiment-price gap ────────────────────────────────────────────
    gap, gap_signal = _compute_sentiment_price_gap(
        consensus_compound, price_change_24h_pct
    )

    # ── 5. Lesson penalty ─────────────────────────────────────────────────
    lessons = get_lessons()
    lesson_penalty = _compute_lesson_penalty(symbol, lessons)
    adjusted = consensus_compound * (1.0 - lesson_penalty)

    # ── 6. Research quality score ─────────────────────────────────────────
    total_items = sum(sb.item_count for sb in source_briefs)
    quality = _assess_quality(len(source_briefs), total_items, agreement)

    # ── 7. Top snippets for logging ───────────────────────────────────────
    all_snippets = []
    for sb in source_briefs:
        all_snippets.extend(sb.top_snippets[:2])

    duration = round(time.time() - start_ts, 2)

    brief = ResearchBrief(
        symbol=symbol,
        timestamp=datetime.utcnow().isoformat(),
        consensus_label=consensus_label,
        consensus_compound=round(consensus_compound, 4),
        consensus_confidence=round(consensus_confidence, 4),
        sources=source_briefs,
        price_change_24h_pct=price_change_24h_pct,
        sentiment_price_gap=round(gap, 4),
        gap_signal=gap_signal,
        source_agreement=round(agreement, 4),
        divergent_sources=divergent,
        lesson_penalty=round(lesson_penalty, 4),
        adjusted_compound=round(adjusted, 4),
        total_items_analysed=total_items,
        duration_seconds=duration,
        research_quality=quality,
    )

    _log.info(
        f"Research complete: {symbol} → {consensus_label.upper()} "
        f"(compound={consensus_compound:+.3f}, gap={gap:+.3f}, quality={quality})",
        data={
            "symbol": symbol,
            "consensus": consensus_label,
            "compound": consensus_compound,
            "gap_signal": gap_signal,
            "agreement": agreement,
            "sources": len(source_briefs),
            "items": total_items,
            "duration_s": duration,
        },
    )

    return brief


async def run_research_batch(
    markets: list,  # List[MarketSnapshot]
    max_concurrent: int = 5,
) -> Dict[str, ResearchBrief]:
    """
    Run research on multiple markets concurrently.
    Throttled to max_concurrent at a time to respect rate limits.
    """
    _log.info(
        f"Research batch started: {len(markets)} markets",
        data={"symbols": [m.symbol for m in markets[:10]]},
    )

    results: Dict[str, ResearchBrief] = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _research_one(market) -> Tuple[str, ResearchBrief]:
        async with semaphore:
            brief = await run_research(
                symbol=market.symbol,
                price_change_24h_pct=market.price_change_pct_24h,
                anomaly_flags=market.anomaly_flags,
            )
            return market.symbol, brief

    tasks = [_research_one(m) for m in markets]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for result in completed:
        if isinstance(result, Exception):
            _log.error(f"Research batch task failed: {result}")
        else:
            sym, brief = result
            results[sym] = brief

    _log.info(
        f"Research batch complete: {len(results)}/{len(markets)} succeeded",
        data={"succeeded": list(results.keys())},
    )
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_source_brief(
    source: str,
    agg: AggregatedSentiment,
    scores: List[SentimentScore],
) -> SourceBrief:
    snippets = [s.text_snippet for s in scores if s.confidence > 0.3][:5]
    return SourceBrief(
        source=source,
        label=agg.label,
        compound=agg.compound_avg,
        confidence=agg.confidence,
        item_count=agg.total_items,
        weight=SOURCE_WEIGHTS.get(source, 0.5),
        top_snippets=snippets,
    )


def _aggregate_cross_source(briefs: List[SourceBrief]) -> dict:
    """Weighted average across sources. Returns consensus dict."""
    total_weight = sum(b.weight for b in briefs)
    if total_weight == 0:
        return {"label": "neutral", "compound": 0.0, "confidence": 0.0,
                "agreement": 0.0, "divergent": []}

    weighted_compound = sum(b.compound * b.weight for b in briefs) / total_weight
    weighted_confidence = sum(b.confidence * b.weight for b in briefs) / total_weight

    # Label by weighted compound
    if weighted_compound >= 0.08:
        label = "bullish"
    elif weighted_compound <= -0.08:
        label = "bearish"
    else:
        label = "neutral"

    # Agreement: how many sources agree with the consensus label
    agreeing = [b for b in briefs if b.label == label]
    agreement = len(agreeing) / len(briefs) if briefs else 0.0
    divergent = [b.source for b in briefs if b.label != label]

    return {
        "label": label,
        "compound": weighted_compound,
        "confidence": weighted_confidence,
        "agreement": agreement,
        "divergent": divergent,
    }


def _compute_sentiment_price_gap(
    compound: float,
    price_change_pct: float,
) -> Tuple[float, str]:
    """
    Compute the gap between sentiment and price movement.
    Large positive gap = sentiment more bullish than price has moved
      → potential upside opportunity (market hasn't priced in the sentiment)
    Large negative gap = price moved up but sentiment is bearish
      → potential mean-reversion / fade signal
    """
    # Normalise price change to [-1, +1] scale (cap at ±20%)
    price_norm = max(-1.0, min(1.0, price_change_pct / 20.0))
    gap = compound - price_norm

    if gap > 0.2:
        signal = "bullish_gap"    # sentiment much more bullish than price action
    elif gap < -0.2:
        signal = "bearish_gap"    # price up but sentiment negative (fade signal)
    elif abs(gap) < 0.1:
        signal = "aligned"        # price and sentiment in sync
    else:
        signal = "neutral"

    return gap, signal


def _compute_lesson_penalty(symbol: str, lessons: dict) -> float:
    """
    Apply a penalty to the adjusted compound score based on past lessons.
    Markets with repeated bad-prediction losses get penalised.
    Returns penalty in [0, 0.5].
    """
    past_lessons = [
        l for l in lessons.get("lessons", [])
        if l.get("symbol") == symbol and l.get("failure_type") == "bad_prediction"
    ]
    # Each bad prediction lesson reduces confidence by 5%, capped at 50%
    penalty = min(len(past_lessons) * 0.05, 0.50)
    return penalty


def _assess_quality(n_sources: int, n_items: int, agreement: float) -> str:
    if n_sources >= 3 and n_items >= 30 and agreement >= 0.7:
        return "high"
    elif n_sources >= 2 and n_items >= 10:
        return "medium"
    return "low"


def _neutral_brief(
    symbol: str,
    price_change: float,
    duration: float,
) -> ResearchBrief:
    return ResearchBrief(
        symbol=symbol,
        timestamp=datetime.utcnow().isoformat(),
        consensus_label="neutral",
        consensus_compound=0.0,
        consensus_confidence=0.0,
        price_change_24h_pct=price_change,
        sentiment_price_gap=0.0,
        gap_signal="neutral",
        source_agreement=0.0,
        lesson_penalty=0.0,
        adjusted_compound=0.0,
        total_items_analysed=0,
        duration_seconds=duration,
        research_quality="low",
    )


# ── Sync wrapper for scheduler ────────────────────────────────────────────────

def run_research_sync(symbol: str, price_change_24h_pct: float = 0.0) -> Optional[ResearchBrief]:
    try:
        return asyncio.run(run_research(symbol, price_change_24h_pct))
    except Exception as e:
        _log.error(f"Research sync failed for {symbol}: {e}")
        return None
