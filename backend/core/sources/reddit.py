"""
sources/reddit.py — Reddit connector.
Fetches posts and top comments from crypto subreddits.
Falls back to mock data if PRAW credentials are not configured.
All content passes through sanitiser before analysis.
"""
import asyncio
import random
from dataclasses import dataclass
from typing import List, Optional
from ..sanitiser import clean, clean_title, is_suspicious
from ..logger import get_logger
from ..config import load_config

_log = get_logger("research")

_MOCK_POSTS = {
    "BTC": [
        ("Bitcoin breaking above 200-day MA, historically very bullish signal", 2840),
        ("Just bought more BTC on this dip, long-term conviction unchanged", 1920),
        ("On-chain analysis: exchange outflows at 6-month high, strong accumulation", 3100),
        ("BTC miners not selling despite price increase, supply squeeze incoming?", 1540),
        ("HODLers not moving coins even with 40% gains, diamond hands confirmed", 2200),
        ("Bitcoin hashrate hits all-time high today, network fundamentals stronger than ever", 1870),
        ("Macro environment turning favorable for Bitcoin as inflation data comes in hot", 980),
        ("Large BTC wallet dormant for 5 years just moved, potential whale activity", 3400),
    ],
    "ETH": [
        ("Ethereum staking now exceeds 30% of supply, supply shock incoming", 2100),
        ("Layer 2 transaction count surpasses Ethereum mainnet, ecosystem scaling", 1650),
        ("ETH/BTC ratio forming potential reversal, altseason signal?", 1320),
        ("Vitalik's latest post suggests major protocol improvements coming", 2800),
        ("DeFi TVL on Ethereum hits new all-time high across major protocols", 1900),
        ("ETH burn rate spiking as NFT activity resurges, deflationary pressure", 1420),
    ],
    "SOL": [
        ("Solana TPS hitting record highs, network performance improved massively", 1800),
        ("SOL staking yield of 7% making it attractive vs traditional finance", 1200),
        ("Major DeFi protocol migrating to Solana for lower fees and faster speed", 2300),
        ("Solana mobile phone gaining traction, crypto adoption mainstream push", 980),
    ],
    "GENERAL": [
        ("Crypto fear and greed index at 72 (greed), market momentum strong", 2400),
        ("DXY weakening again, historically correlates with crypto upside moves", 1850),
        ("Fed pivot expectations growing, risk assets including crypto benefiting", 3200),
        ("Total crypto market cap holding above key support at $2.4T, bulls intact", 1600),
        ("Stablecoin supply growing rapidly, dry powder accumulating for buys", 1980),
        ("Exchange reserves at multi-year lows, supply side looking very tight", 2750),
        ("Bitcoin ETF inflows continue to accelerate, institutional buying real", 3100),
        ("Altcoins starting to outperform BTC, rotation into higher beta assets", 1420),
        ("Crypto VC funding picking up after bear market, projects being built", 890),
        ("Options market pricing in significant upside, skew turning positive", 1670),
    ],
}

_BEARISH_MOCK = {
    "GENERAL": [
        ("Crypto regulations tightening in multiple jurisdictions, headwinds ahead", 1900),
        ("Leverage in crypto market at dangerous levels, potential for cascade liquidations", 2200),
        ("Bitcoin dominance rising sharply, funds rotating out of altcoins to safety", 1650),
        ("Macro headwinds persist as Fed signals rates higher for longer", 2800),
        ("Large crypto fund rumored to be in trouble, contagion risk elevated", 3400),
        ("On-chain analysis shows distribution from long-term holders at current prices", 1780),
    ],
}


@dataclass
class RedditPost:
    title: str
    body: str
    score: int
    subreddit: str
    combined_text: str
    url: str = ""


async def fetch_reddit(
    subreddits: List[str],
    symbol: str,
    max_posts: int = 20,
) -> List[RedditPost]:
    """
    Fetch relevant posts from crypto subreddits.
    Tries PRAW first; falls back to mock data if not configured.
    """
    cfg = load_config()
    reddit_cfg = cfg.get("reddit", {})
    client_id = reddit_cfg.get("client_id", "")
    client_secret = reddit_cfg.get("client_secret", "")

    if client_id and client_secret:
        posts = await _fetch_via_praw(subreddits, symbol, client_id, client_secret, max_posts)
        if posts:
            return posts

    _log.warning(f"Reddit: using mock data for {symbol} (no credentials configured)")
    return _get_mock_posts(symbol, max_posts)


async def _fetch_via_praw(
    subreddits: List[str],
    symbol: str,
    client_id: str,
    client_secret: str,
    max_posts: int,
) -> List[RedditPost]:
    """Fetch real Reddit data via PRAW (runs in thread pool to avoid blocking)."""
    def _sync_fetch():
        try:
            import praw
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="CryptoSignalBot/1.0",
            )
            base = symbol.split("/")[0].upper()
            posts = []
            for sub_name in subreddits[:3]:
                try:
                    sub = reddit.subreddit(sub_name)
                    for post in sub.hot(limit=25):
                        title = clean_title(post.title)
                        body = clean(post.selftext or "")
                        if is_suspicious(title) or is_suspicious(body):
                            continue
                        combined = f"{title}. {body}".strip()
                        if base.lower() in combined.lower() or _is_general_crypto(combined):
                            posts.append(RedditPost(
                                title=title,
                                body=body[:500],
                                score=post.score,
                                subreddit=sub_name,
                                combined_text=combined,
                                url=f"https://reddit.com{post.permalink}",
                            ))
                except Exception as e:
                    _log.warning(f"Reddit subreddit {sub_name} failed: {e}")
            return posts[:max_posts]
        except Exception as e:
            _log.warning(f"PRAW fetch failed: {e}")
            return []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_fetch)


def _is_general_crypto(text: str) -> bool:
    terms = {"crypto", "bitcoin", "btc", "ethereum", "eth", "altcoin",
             "market", "bull", "bear", "defi", "blockchain"}
    text_lower = text.lower()
    return any(t in text_lower for t in terms)


def _get_mock_posts(symbol: str, max_posts: int) -> List[RedditPost]:
    base = symbol.split("/")[0].upper()
    pool = list(_MOCK_POSTS.get(base, [])) + list(_MOCK_POSTS["GENERAL"])

    # Occasionally mix in some bearish posts (30% chance) for realism
    if random.random() < 0.3:
        pool += _BEARISH_MOCK["GENERAL"]

    random.shuffle(pool)
    return [
        RedditPost(
            title=title,
            body="",
            score=score,
            subreddit="CryptoCurrency",
            combined_text=title,
            url="",
        )
        for title, score in pool[:max_posts]
    ]
