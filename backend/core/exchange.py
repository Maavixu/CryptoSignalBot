"""
exchange.py — Unified exchange adapter built on ccxt.
All scanner, risk, and execution modules use this.
Never call ccxt directly elsewhere.
"""
import asyncio
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    import ccxt.async_support as ccxt_async
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from .config import load_config
from .logger import get_logger

_log = get_logger("exchange")


class ExchangeAdapter:
    """
    Wraps ccxt to provide a clean interface.
    Falls back to mock data if ccxt is unavailable or exchange unreachable.
    """

    def __init__(self, exchange_id: str = "binance", use_mock: bool = False):
        self.exchange_id = exchange_id
        self.use_mock = use_mock or not CCXT_AVAILABLE
        self._exchange = None
        self._exchange_sync = None
        self._connected = False

        if not self.use_mock and CCXT_AVAILABLE:
            try:
                ExClass = getattr(ccxt_async, exchange_id)
                self._exchange = ExClass({
                    "enableRateLimit": True,
                    "timeout": 30000,
                })
                ExClassSync = getattr(ccxt, exchange_id)
                self._exchange_sync = ExClassSync({
                    "enableRateLimit": True,
                    "timeout": 30000,
                })
                self._connected = True
                _log.info(f"Exchange adapter initialised: {exchange_id}")
            except Exception as e:
                _log.warning(f"Failed to init {exchange_id}, using mock: {e}")
                self.use_mock = True

    async def close(self):
        if self._exchange:
            await self._exchange.close()

    # ── Market data ──────────────────────────────────────────────────────────

    async def fetch_tickers(self, symbols: Optional[List[str]] = None) -> Dict[str, dict]:
        """Fetch ticker data for all or specified symbols."""
        if self.use_mock:
            return self._mock_tickers()
        try:
            tickers = await self._exchange.fetch_tickers(symbols)
            _log.debug(f"Fetched {len(tickers)} tickers")
            return tickers
        except Exception as e:
            _log.error(f"fetch_tickers failed: {e}")
            return self._mock_tickers()

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """Fetch order book for spread and depth calculation."""
        if self.use_mock:
            return self._mock_order_book(symbol)
        try:
            ob = await self._exchange.fetch_order_book(symbol, limit)
            return ob
        except Exception as e:
            _log.error(f"fetch_order_book({symbol}) failed: {e}")
            return self._mock_order_book(symbol)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 200,
        since: Optional[int] = None,
    ) -> List[List]:
        """
        Fetch OHLCV candles. Paginates automatically for large requests.
        Returns list of [timestamp, open, high, low, close, volume].
        """
        if self.use_mock:
            return self._mock_ohlcv(symbol, limit)
        try:
            all_candles = []
            fetch_since = since

            while True:
                candles = await self._exchange.fetch_ohlcv(
                    symbol, timeframe, since=fetch_since, limit=min(limit, 1000)
                )
                if not candles:
                    break
                all_candles.extend(candles)
                if len(all_candles) >= limit or len(candles) < 1000:
                    break
                # Advance cursor
                fetch_since = candles[-1][0] + 1
                await asyncio.sleep(0.2)  # respect rate limit

            return all_candles[:limit]
        except Exception as e:
            _log.error(f"fetch_ohlcv({symbol}, {timeframe}) failed: {e}")
            return self._mock_ohlcv(symbol, limit)

    async def get_markets(self) -> Dict[str, dict]:
        """Fetch all available markets."""
        if self.use_mock:
            return self._mock_markets()
        try:
            markets = await self._exchange.load_markets()
            return markets
        except Exception as e:
            _log.error(f"get_markets failed: {e}")
            return self._mock_markets()

    def get_spread_pct(self, order_book: dict) -> float:
        """Calculate bid-ask spread as percentage."""
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if not bids or not asks:
            return 999.0
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2
        if mid == 0:
            return 999.0
        return ((best_ask - best_bid) / mid) * 100

    def get_order_book_depth_usd(self, order_book: dict, levels: int = 10) -> float:
        """Calculate total USD depth across N levels on each side."""
        bids = order_book.get("bids", [])[:levels]
        asks = order_book.get("asks", [])[:levels]
        bid_depth = sum(price * qty for price, qty in bids)
        ask_depth = sum(price * qty for price, qty in asks)
        return (bid_depth + ask_depth) / 2

    # ── Mock data (for demo / offline dev) ───────────────────────────────────

    def _mock_tickers(self) -> Dict[str, dict]:
        import random
        symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
            "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "LINK/USDT", "UNI/USDT",
            "ATOM/USDT", "LTC/USDT", "XRP/USDT", "DOGE/USDT", "SHIB/USDT",
            "APT/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT", "SUI/USDT",
        ]
        tickers = {}
        base_prices = {
            "BTC/USDT": 67000, "ETH/USDT": 3500, "BNB/USDT": 580,
            "SOL/USDT": 180, "ADA/USDT": 0.65, "AVAX/USDT": 38,
            "DOT/USDT": 9.5, "MATIC/USDT": 1.1, "LINK/USDT": 18,
            "UNI/USDT": 12, "ATOM/USDT": 10, "LTC/USDT": 95,
            "XRP/USDT": 0.62, "DOGE/USDT": 0.18, "SHIB/USDT": 0.000028,
            "APT/USDT": 12, "ARB/USDT": 1.8, "OP/USDT": 3.2,
            "INJ/USDT": 35, "SUI/USDT": 1.9,
        }
        for sym in symbols:
            base = base_prices.get(sym, 1.0)
            change_pct = random.uniform(-8, 8)
            last = base * (1 + change_pct / 100)
            vol_base = random.uniform(5e6, 5e8)
            tickers[sym] = {
                "symbol": sym,
                "last": last,
                "bid": last * 0.9995,
                "ask": last * 1.0005,
                "high": last * random.uniform(1.01, 1.08),
                "low": last * random.uniform(0.92, 0.99),
                "baseVolume": vol_base / last,
                "quoteVolume": vol_base,
                "percentage": change_pct,
                "timestamp": int(time.time() * 1000),
            }
        return tickers

    def _mock_order_book(self, symbol: str) -> dict:
        import random
        base_prices = {
            "BTC/USDT": 67000, "ETH/USDT": 3500, "BNB/USDT": 580,
            "SOL/USDT": 180, "ADA/USDT": 0.65, "AVAX/USDT": 38,
            "DOT/USDT": 9.5, "MATIC/USDT": 1.1, "LINK/USDT": 18,
            "UNI/USDT": 12, "ATOM/USDT": 10, "LTC/USDT": 95,
            "XRP/USDT": 0.62, "DOGE/USDT": 0.18, "SHIB/USDT": 0.000028,
            "APT/USDT": 12, "ARB/USDT": 1.8, "OP/USDT": 3.2,
            "INJ/USDT": 35, "SUI/USDT": 1.9,
        }
        mid = base_prices.get(symbol, 10.0)
        spread_pct = random.uniform(0.02, 0.35)
        spread = mid * spread_pct / 100
        # Qty sized to produce realistic USD depth ($50K-$2M per side)
        target_depth_usd = random.uniform(80_000, 1_500_000)
        qty_per_level = (target_depth_usd / 10) / mid
        return {
            "symbol": symbol,
            "bids": [[mid - spread/2 - i*(mid*0.001), qty_per_level * random.uniform(0.5, 1.5)] for i in range(20)],
            "asks": [[mid + spread/2 + i*(mid*0.001), qty_per_level * random.uniform(0.5, 1.5)] for i in range(20)],
            "timestamp": int(time.time() * 1000),
        }

    def _mock_ohlcv(self, symbol: str, limit: int = 200) -> List[List]:
        import random
        import math
        candles = []
        now = int(time.time() * 1000)
        price = 100.0
        for i in range(limit):
            ts = now - (limit - i) * 3600 * 1000
            change = random.gauss(0, 0.02)
            open_ = price
            close = price * (1 + change)
            high = max(open_, close) * random.uniform(1.001, 1.015)
            low = min(open_, close) * random.uniform(0.985, 0.999)
            vol = random.uniform(1e5, 1e7)
            candles.append([ts, open_, high, low, close, vol])
            price = close
        return candles

    def _mock_markets(self) -> Dict[str, dict]:
        tickers = self._mock_tickers()
        return {sym: {"symbol": sym, "active": True, "quote": "USDT"}
                for sym in tickers}


# ── Singleton factory ─────────────────────────────────────────────────────────

_adapter: Optional[ExchangeAdapter] = None


def get_adapter(force_mock: bool = False) -> ExchangeAdapter:
    global _adapter
    if _adapter is None:
        cfg = load_config()
        exchange_id = cfg.get("exchange", {}).get("id", "binance")
        mode = cfg.get("bot", {}).get("mode", "paper")
        use_mock = force_mock or mode == "paper"
        _adapter = ExchangeAdapter(exchange_id=exchange_id, use_mock=use_mock)
    return _adapter
