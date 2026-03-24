"""
sentiment.py — NLP sentiment classification for crypto content.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) as the
primary classifier — well-suited to short, informal social/news text.

Outputs a SentimentScore with:
  - label: "bullish" | "bearish" | "neutral"
  - compound: float in [-1, +1]
  - confidence: float in [0, 1]
  - components: raw positive / negative / neutral breakdown
"""
from dataclasses import dataclass, field
from typing import List, Optional
import statistics

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Crypto-specific lexicon additions — VADER doesn't know these terms
CRYPTO_LEXICON = {
    # Bullish terms
    "bullish": 2.5, "moon": 2.0, "mooning": 2.5, "hodl": 1.5,
    "hodling": 1.5, "accumulate": 1.5, "accumulating": 1.5,
    "breakout": 2.0, "ath": 2.5, "all-time high": 2.5,
    "parabolic": 2.0, "ripping": 2.0, "pumping": 1.8, "surge": 1.8,
    "rally": 1.8, "rallying": 1.8, "buy the dip": 2.0,
    "institutional buying": 2.0, "adoption": 1.5, "mainstream": 1.2,
    "halvening": 1.5, "halving": 1.5, "deflationary": 1.2,
    "undervalued": 1.8, "support": 1.0, "bouncing": 1.5,

    # Bearish terms
    "bearish": -2.5, "dump": -2.0, "dumping": -2.5, "rekt": -3.0,
    "liquidated": -2.5, "liquidation": -2.5, "crash": -2.5,
    "crashing": -2.5, "collapse": -2.5, "collapsing": -2.5,
    "rugpull": -3.0, "rug pull": -3.0, "scam": -2.5,
    "exit scam": -3.0, "ponzi": -3.0, "dead": -2.0, "obituary": -2.5,
    "overvalued": -1.8, "resistance": -0.8, "rejected": -1.5,
    "distribution": -1.2, "bleeding": -2.0, "capitulation": -2.5,
    "fear": -1.5, "panic": -2.0, "panicking": -2.5,
    "sell the news": -1.5, "short": -1.0, "shorting": -1.5,

    # Neutral / mixed (small adjustments)
    "volatile": -0.3, "volatility": -0.2, "correction": -0.8,
    "consolidating": 0.2, "sideways": 0.0, "choppy": -0.3,
}


@dataclass
class SentimentScore:
    label: str              # "bullish" | "bearish" | "neutral"
    compound: float         # [-1, +1]
    confidence: float       # [0, 1] — how decisive the signal is
    positive: float
    negative: float
    neutral: float
    text_snippet: str = ""  # first 80 chars of source text


@dataclass
class AggregatedSentiment:
    """Combined sentiment across multiple text items from one source."""
    source: str
    symbol: str
    label: str              # majority vote label
    compound_avg: float
    confidence: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_items: int
    scores: List[SentimentScore] = field(default_factory=list)


# ── Analyser singleton ────────────────────────────────────────────────────────

class CryptoSentimentAnalyser:
    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()
        # Inject crypto lexicon
        self._vader.lexicon.update(CRYPTO_LEXICON)

    def score(self, text: str) -> SentimentScore:
        """
        Score a single piece of text.
        Returns SentimentScore with label, compound, and confidence.
        """
        if not text or len(text.strip()) < 5:
            return SentimentScore("neutral", 0.0, 0.0, 0.0, 0.0, 1.0)

        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]

        # Label thresholds tuned for crypto content (slightly stricter than default)
        if compound >= 0.12:
            label = "bullish"
        elif compound <= -0.12:
            label = "bearish"
        else:
            label = "neutral"

        # Confidence: distance from zero, scaled to [0,1]
        # A compound of ±0.5 → confidence ~0.5; ±1.0 → ~1.0
        confidence = min(abs(compound) * 1.5, 1.0)

        return SentimentScore(
            label=label,
            compound=round(compound, 4),
            confidence=round(confidence, 4),
            positive=round(scores["pos"], 4),
            negative=round(scores["neg"], 4),
            neutral=round(scores["neu"], 4),
            text_snippet=text[:80],
        )

    def score_batch(self, texts: List[str]) -> List[SentimentScore]:
        return [self.score(t) for t in texts if t]

    def aggregate(
        self,
        scores: List[SentimentScore],
        source: str,
        symbol: str,
        min_confidence: float = 0.0,
    ) -> AggregatedSentiment:
        """
        Aggregate a list of SentimentScores into one source-level signal.
        Low-confidence items are down-weighted in the compound average.
        """
        if not scores:
            return AggregatedSentiment(
                source=source, symbol=symbol, label="neutral",
                compound_avg=0.0, confidence=0.0,
                bullish_count=0, bearish_count=0, neutral_count=0,
                total_items=0,
            )

        # Filter by minimum confidence
        confident = [s for s in scores if s.confidence >= min_confidence] or scores

        # Weighted average compound (weight = confidence)
        weights = [s.confidence for s in confident]
        total_w = sum(weights) or 1.0
        compound_avg = sum(s.compound * s.confidence for s in confident) / total_w

        # Counts
        bullish = sum(1 for s in confident if s.label == "bullish")
        bearish = sum(1 for s in confident if s.label == "bearish")
        neutral = sum(1 for s in confident if s.label == "neutral")

        # Majority label
        counts = {"bullish": bullish, "bearish": bearish, "neutral": neutral}
        label = max(counts, key=counts.get)

        # Aggregate confidence: how much agreement exists
        majority_pct = counts[label] / len(confident)
        agg_confidence = round(majority_pct * min(abs(compound_avg) * 2, 1.0), 4)

        return AggregatedSentiment(
            source=source,
            symbol=symbol,
            label=label,
            compound_avg=round(compound_avg, 4),
            confidence=agg_confidence,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            total_items=len(scores),
            scores=confident[:10],  # keep top 10 for logging
        )


# Singleton
_analyser: Optional[CryptoSentimentAnalyser] = None


def get_analyser() -> CryptoSentimentAnalyser:
    global _analyser
    if _analyser is None:
        _analyser = CryptoSentimentAnalyser()
    return _analyser
