"""
Market Data Service v2
======================
Scans top 25 coins by volume on Kraken.
"""

import ccxt
import pandas as pd
import logging
from typing import List, Optional
from datetime import datetime, timezone

log = logging.getLogger(__name__)

SKIP_BASES = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "PYUSD", "FDUSD", "EUR", "GBP"}


class MarketDataService:
    def __init__(self, quote: str = "USD", universe_size: int = 25):
        self.quote = quote
        self.universe_size = universe_size
        self.exchange = ccxt.kraken({"enableRateLimit": True, "timeout": 30000})
        self._universe: List[str] = []
        self._last_universe_refresh: Optional[datetime] = None

    def get_universe(self, force: bool = False) -> List[str]:
        """Get top coins ranked by 24h volume. Refreshes every 4 hours."""
        now = datetime.now(timezone.utc)
        if self._universe and not force and self._last_universe_refresh:
            elapsed = (now - self._last_universe_refresh).total_seconds()
            if elapsed < 4 * 3600:
                return self._universe

        try:
            markets = self.exchange.load_markets()
            pairs = [
                s.replace(f"/{self.quote}", "")
                for s in markets
                if s.endswith(f"/{self.quote}")
                and markets[s].get("active", True)
                and s.split("/")[0] not in SKIP_BASES
            ]

            # Fetch tickers for volume ranking
            try:
                tickers = self.exchange.fetch_tickers()
                ranked = sorted(
                    [(s, tickers.get(f"{s}/{self.quote}", {}).get("quoteVolume", 0) or 0)
                     for s in pairs],
                    key=lambda x: x[1], reverse=True,
                )
                self._universe = [s for s, v in ranked[:self.universe_size] if v > 0]
            except Exception:
                self._universe = pairs[:self.universe_size]

            self._last_universe_refresh = now
            log.info(f"📡 Universe: {len(self._universe)} coins — "
                     f"top 5: {', '.join(self._universe[:5])}")
            return self._universe

        except Exception as e:
            log.error(f"Failed to load universe: {e}")
            return self._universe or ["BTC", "ETH", "SOL", "XRP", "DOGE"]

    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candles for a symbol."""
        pair = f"{symbol}/{self.quote}"
        try:
            ohlcv = self.exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df
        except Exception as e:
            log.debug(f"Failed {pair} {timeframe}: {e}")
            return pd.DataFrame()

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        pair = f"{symbol}/{self.quote}"
        try:
            ticker = self.exchange.fetch_ticker(pair)
            return float(ticker.get("last", 0))
        except Exception:
            return None
