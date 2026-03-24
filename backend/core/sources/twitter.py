"""
sources/twitter.py — Twitter/X connector.
Real Twitter API (v2 Bearer Token) if credentials are present;
otherwise returns realistic mock tweet data.
All content sanitised before analysis.
"""
import asyncio
import random
from dataclasses import dataclass
from typing import List
import aiohttp

from ..sanitiser import clean, clean_title, is_suspicious
from ..logger import get_logger
from ..config import load_config

_log = get_logger("research")

_BULLISH_TWEETS = {
    "BTC": [
        "Bitcoin printing the most beautiful weekly candle structure I've seen in years 🔥",
        "BTC whales buying every single dip. On-chain data doesn't lie. Accumulation phase.",
        "The macro setup for Bitcoin right now is better than pre-2020 bull run. HODL.",
        "Bitcoin hashrate at ATH. Miners are extremely confident. Bullish af.",
        "Just checked BTC exchange reserves. LOWEST in 5 years. Supply shock incoming.",
        "Institutional BTC buying through ETFs hasn't slowed down once. Follow the money.",
        "Bitcoin 200-week MA holding perfectly. Long-term holders not selling. Diamond hands.",
    ],
    "ETH": [
        "Ethereum staking ratio hitting 30%. Supply getting locked up fast. Bullish.",
        "ETH L2 ecosystem exploding. Arbitrum + Optimism daily txs now beating mainnet.",
        "If you don't understand why ETH is the most undervalued asset rn you're not paying attention",
        "ETH burn rate spiking. Deflation kicking in hard. Price has to catch up eventually.",
        "Ethereum TVL recovering fast. DeFi summer 2.0 is coming and ETH leads it.",
    ],
    "SOL": [
        "Solana is officially back. No more outages. TPS at records. Dev activity insane.",
        "SOL ecosystem growing faster than any L1 I've tracked. Bullish long term.",
        "Solana phone actually selling out. Crypto mobile is happening and SOL owns it.",
    ],
    "GENERAL": [
        "Crypto fear & greed just flipped to greed. Early innings of a move up.",
        "DXY rolling over again. Every time this happens crypto pumps hard. Watching closely.",
        "Total3 (alts ex BTC/ETH) showing massive cup and handle. Altseason loading.",
        "Stablecoin supply on exchanges at record high. That's future buying pressure.",
        "Fed pivot incoming. Risk assets including crypto will moon when they announce it.",
        "Crypto adoption curve looks identical to internet in 1997. We're early.",
        "Smart money rotating INTO crypto right now. ETF flows confirm this.",
        "When normies panic sell, whales accumulate. We've seen this movie before.",
    ],
}

_BEARISH_TWEETS = {
    "GENERAL": [
        "This rally looks like classic distribution to me. Getting ready to short.",
        "Too much leverage in the system right now. One bad day and it all cascades.",
        "Regulatory crackdown getting serious. Compliance costs will kill smaller projects.",
        "Crypto sentiment too bullish. Contrarian signal to reduce exposure here.",
        "Bitcoin dominance rising = altcoins about to get crushed. Be careful out there.",
        "Exchange hacks becoming more frequent. Security concerns are real risk.",
        "Macro still not clear. Rates staying higher for longer = bad for risk assets.",
    ],
}

_NEUTRAL_TWEETS = {
    "GENERAL": [
        "Bitcoin moving sideways for 3 weeks. Need a catalyst in either direction.",
        "Watching $BTC closely at this key resistance. Either breaks or rejects hard.",
        "Market structure neutral here. Could go either way. Waiting for confirmation.",
        "Volume drying up on crypto. Consolidation phase. Patient traders win.",
        "Not adding or reducing here. Just watching the charts. No clear signal.",
    ],
}


@dataclass
class Tweet:
    text: str
    like_count: int
    retweet_count: int
    author_followers: int
    combined_text: str
    weight: float  # influence weight based on engagement


async def fetch_twitter(
    symbol: str,
    max_tweets: int = 30,
) -> List[Tweet]:
    """
    Fetch tweets about the symbol.
    Tries real Twitter API v2 if bearer token configured;
    otherwise returns weighted mock tweets.
    """
    cfg = load_config()
    bearer = cfg.get("twitter", {}).get("bearer_token", "")

    if bearer:
        tweets = await _fetch_real(bearer, symbol, max_tweets)
        if tweets:
            return tweets

    _log.warning(f"Twitter: using mock data for {symbol}")
    return _get_mock_tweets(symbol, max_tweets)


async def _fetch_real(bearer: str, symbol: str, max_tweets: int) -> List[Tweet]:
    """Fetch real tweets via Twitter API v2."""
    base = symbol.split("/")[0]
    query = f"#{base} OR #{base}USDT crypto -is:retweet lang:en"
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": min(max_tweets, 100),
        "tweet.fields": "public_metrics,author_id",
        "expansions": "author_id",
        "user.fields": "public_metrics",
    }
    headers = {"Authorization": f"Bearer {bearer}"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    _log.warning(f"Twitter API returned {resp.status}")
                    return []
                data = await resp.json()

        tweets = []
        for t in data.get("data", []):
            text = clean(t.get("text", ""))
            if is_suspicious(text) or not text:
                continue
            metrics = t.get("public_metrics", {})
            likes = metrics.get("like_count", 0)
            rts = metrics.get("retweet_count", 0)
            weight = 1.0 + (likes + rts * 2) / 1000
            tweets.append(Tweet(
                text=text, like_count=likes, retweet_count=rts,
                author_followers=0, combined_text=text, weight=min(weight, 5.0),
            ))
        return tweets

    except Exception as e:
        _log.warning(f"Twitter real fetch failed: {e}")
        return []


def _get_mock_tweets(symbol: str, max_tweets: int) -> List[Tweet]:
    base = symbol.split("/")[0].upper()

    # Build pool: 60% bullish, 20% bearish, 20% neutral (configurable realism)
    bullish = _BULLISH_TWEETS.get(base, []) + _BULLISH_TWEETS["GENERAL"]
    bearish = _BEARISH_TWEETS["GENERAL"]
    neutral = _NEUTRAL_TWEETS["GENERAL"]

    pool = []
    random.shuffle(bullish)
    random.shuffle(bearish)
    random.shuffle(neutral)

    b_count = int(max_tweets * 0.60)
    bear_count = int(max_tweets * 0.20)
    n_count = max_tweets - b_count - bear_count

    for text in bullish[:b_count]:
        likes = random.randint(50, 5000)
        rts = random.randint(10, 1000)
        pool.append(Tweet(
            text=text, like_count=likes, retweet_count=rts,
            author_followers=random.randint(1000, 500000),
            combined_text=text,
            weight=1.0 + (likes + rts * 2) / 1000,
        ))
    for text in bearish[:bear_count]:
        likes = random.randint(20, 2000)
        rts = random.randint(5, 500)
        pool.append(Tweet(
            text=text, like_count=likes, retweet_count=rts,
            author_followers=random.randint(500, 100000),
            combined_text=text,
            weight=1.0 + (likes + rts * 2) / 1000,
        ))
    for text in neutral[:n_count]:
        likes = random.randint(10, 800)
        rts = random.randint(2, 200)
        pool.append(Tweet(
            text=text, like_count=likes, retweet_count=rts,
            author_followers=random.randint(500, 50000),
            combined_text=text,
            weight=1.0 + (likes + rts * 2) / 1000,
        ))

    random.shuffle(pool)
    return pool[:max_tweets]
