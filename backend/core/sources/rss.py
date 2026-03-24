"""
sources/rss.py — RSS/Atom feed connector.
Fetches crypto news from CryptoPanic, CoinDesk, and other RSS feeds.
All content is sanitised before returning.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional
import feedparser
import aiohttp

from ..sanitiser import clean, clean_title, is_suspicious
from ..logger import get_logger

_log = get_logger("research")

TIMEOUT_SECONDS = 8
MAX_RETRIES = 2

# Fallback mock articles if all feeds fail (for demo / offline dev)
_MOCK_ARTICLES = {
    "BTC": [
        "Bitcoin surges past key resistance level as institutional buyers accumulate",
        "BTC whale wallets showing strong accumulation pattern this week",
        "Bitcoin mining difficulty hits all-time high, bullish sign for long-term holders",
        "Major investment firm adds Bitcoin to balance sheet, moon incoming",
        "Bitcoin options market showing bullish skew, traders expect continuation",
        "BTC dominance rising as altcoins bleed against king crypto",
        "On-chain data reveals long-term holders are not selling their Bitcoin",
    ],
    "ETH": [
        "Ethereum network activity surges as DeFi TVL hits new highs",
        "ETH staking yields attracting institutional capital, bullish for price",
        "Ethereum developers announce major upgrade timeline ahead of schedule",
        "ETH burn rate accelerating, deflationary pressure increasing",
        "Layer 2 adoption driving Ethereum ecosystem growth significantly",
        "Ethereum whale accumulation detected on major exchanges this week",
    ],
    "SOL": [
        "Solana DeFi ecosystem recovering strongly after previous network outage",
        "SOL price consolidating at key support, bulls defending $150 level",
        "Solana NFT volumes spike as new collections drive network activity",
        "Institutional interest in Solana growing as ecosystem matures rapidly",
        "Solana developer activity at all-time high according to on-chain metrics",
    ],
    "DEFAULT": [
        "Crypto market showing resilience amid macro uncertainty, bulls hold ground",
        "Altcoin season indicators flashing as Bitcoin dominance falls below 50%",
        "DeFi protocols reporting record revenues as on-chain activity grows",
        "Crypto sentiment index recovering from extreme fear to neutral territory",
        "Market analysts see potential for broad altcoin rally in coming weeks",
        "Exchange outflows accelerating as long-term holders accumulate dips",
        "Derivatives market shows reduced short positioning, potential squeeze ahead",
        "Stablecoin inflows to exchanges increasing, potential buying pressure building",
    ],
}


@dataclass
class RSSArticle:
    title: str
    summary: str
    link: str
    published: str
    source: str
    combined_text: str  # title + summary, sanitised


async def _fetch_feed(session: aiohttp.ClientSession, url: str) -> List[RSSArticle]:
    """Fetch a single RSS feed with retry."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)) as resp:
                if resp.status != 200:
                    _log.warning(f"RSS feed returned {resp.status}: {url}")
                    return []
                content = await resp.text()

            feed = feedparser.parse(content)
            articles = []
            for entry in feed.entries[:20]:
                title = clean_title(entry.get("title", ""))
                summary = clean(entry.get("summary", entry.get("description", "")))

                if is_suspicious(title) or is_suspicious(summary):
                    _log.warning(f"Suspicious content filtered from {url}")
                    continue

                combined = f"{title}. {summary}".strip()
                articles.append(RSSArticle(
                    title=title,
                    summary=summary[:500],
                    link=entry.get("link", ""),
                    published=entry.get("published", ""),
                    source=url,
                    combined_text=combined,
                ))
            return articles

        except asyncio.TimeoutError:
            _log.warning(f"RSS timeout ({attempt+1}/{MAX_RETRIES+1}): {url}")
        except Exception as e:
            _log.warning(f"RSS fetch error ({attempt+1}/{MAX_RETRIES+1}) {url}: {e}")

        if attempt < MAX_RETRIES:
            await asyncio.sleep(1.5 ** attempt)

    return []


async def fetch_rss(
    feeds: List[str],
    symbol: str,
    max_per_source: int = 20,
) -> List[RSSArticle]:
    """
    Fetch and filter articles from multiple RSS feeds.
    Filters by relevance to the symbol (base asset, e.g. BTC from BTC/USDT).
    Falls back to mock data if all feeds fail.
    """
    base = symbol.split("/")[0].upper()

    connector = aiohttp.TCPConnector(limit=10, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_fetch_feed(session, url) for url in feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_articles: List[RSSArticle] = []
    for r in results:
        if isinstance(r, list):
            all_articles.extend(r)

    # Filter by relevance to the symbol
    relevant = _filter_relevant(all_articles, base)

    if not relevant:
        _log.warning(f"No RSS articles found for {symbol}, using mock data")
        relevant = _get_mock_articles(base, symbol)

    _log.info(
        f"RSS: {len(relevant)} articles for {symbol}",
        data={"symbol": symbol, "total_fetched": len(all_articles), "relevant": len(relevant)},
    )
    return relevant[:max_per_source]


def _filter_relevant(articles: List[RSSArticle], base: str) -> List[RSSArticle]:
    """Keep articles that mention the base asset or general crypto market."""
    general_terms = {"crypto", "bitcoin", "btc", "ethereum", "eth", "blockchain",
                     "defi", "nft", "altcoin", "market", "bull", "bear"}
    base_lower = base.lower()

    relevant = []
    for a in articles:
        text_lower = a.combined_text.lower()
        if base_lower in text_lower:
            relevant.append(a)
        elif any(t in text_lower for t in general_terms):
            # General crypto news — weighted lower in aggregation
            relevant.append(a)
    return relevant


def _get_mock_articles(base: str, symbol: str) -> List[RSSArticle]:
    """Generate realistic mock articles for demo / offline mode."""
    import random
    texts = _MOCK_ARTICLES.get(base, _MOCK_ARTICLES["DEFAULT"])
    random.shuffle(texts)
    return [
        RSSArticle(
            title=t,
            summary="",
            link=f"https://example.com/news/{i}",
            published="",
            source="mock_rss",
            combined_text=t,
        )
        for i, t in enumerate(texts[:10])
    ]
