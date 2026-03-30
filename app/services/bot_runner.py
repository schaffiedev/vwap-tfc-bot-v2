"""
Bot Runner v2
=============
Scans top 25 coins during session windows.
Multi-position with Sniper V4 exit system.
"""

import logging
import os
from datetime import datetime, timezone
from typing import List

from app.services.market_data import MarketDataService
from app.core.strategy import (
    score_tfc_setup, is_session_active, get_current_session,
    Trend, SessionWindow,
)
from app.core.paper_trader import PaperAccount

log = logging.getLogger(__name__)


class BotRunner:
    def __init__(self):
        self.quote = os.getenv("QUOTE", "USD")
        self.universe_size = int(os.getenv("UNIVERSE_SIZE", "25"))
        self.market = MarketDataService(quote=self.quote, universe_size=self.universe_size)
        self.account = PaperAccount()
        self._last_run_log: dict = {}

    def run_cycle(self) -> dict:
        """Run one full scan + manage cycle."""
        now = datetime.now(timezone.utc)
        session = get_current_session(now)
        result = {
            "timestamp": now.isoformat(),
            "session": str(session.value),
            "action": "none",
            "scanned": 0,
            "candidates": 0,
            "new_trades": [],
            "closed_trades": [],
            "error": None,
        }

        try:
            # ── 1. MANAGE EXISTING POSITIONS (always, regardless of session) ──
            for symbol in list(self.account.positions.keys()):
                price = self.market.get_price(symbol)
                if price is None:
                    continue
                trade_result = self.account.update_position(symbol, price)
                if trade_result and isinstance(trade_result, dict):
                    result["closed_trades"].append(trade_result)

            # ── 2. SCAN FOR NEW ENTRIES (only during sessions) ──
            if session == SessionWindow.CLOSED:
                result["action"] = "session_closed"
                log.info(f"⏸️ Session closed (next: check at next interval)")
                self._last_run_log = result
                return result

            if not self.account.can_trade:
                result["action"] = "no_slots_or_limit"
                log.info(f"⏸️ Cannot trade: {self.account.open_slots} slots, "
                         f"{self.account.trades_today} trades today")
                self._last_run_log = result
                return result

            # Get universe
            universe = self.market.get_universe()
            if not universe:
                result["error"] = "No universe loaded"
                self._last_run_log = result
                return result

            # Scan all coins
            log.info(f"🔍 Scanning {len(universe)} coins ({session.value})...")
            candidates = []

            for symbol in universe:
                if symbol in self.account.positions:
                    continue  # Already holding

                df = self.market.fetch_ohlcv(symbol, "5m", limit=100)
                if len(df) < 55:
                    continue

                result["scanned"] += 1
                scan = score_tfc_setup(df)
                if scan is not None:
                    scan.symbol = symbol
                    candidates.append(scan)

            candidates.sort(key=lambda x: x.score, reverse=True)
            result["candidates"] = len(candidates)

            if candidates:
                top_names = [f"{c.symbol}({c.score:.2f})" for c in candidates[:5]]
                log.info(f"  Found {len(candidates)} setups. Top: {', '.join(top_names)}")

            # ── 3. FILL OPEN SLOTS ──
            slots_to_fill = self.account.open_slots
            filled = 0

            for scan in candidates[:slots_to_fill]:
                # Calculate SL from ATR
                atr_sl = scan.price * max(scan.atr_pct * 2, 0.025)  # Min 2.5%
                atr_sl = min(atr_sl, scan.price * 0.04)  # Max 4%

                if scan.trend == Trend.BULLISH:
                    stop = scan.price - atr_sl
                    target = scan.price + (atr_sl * 2.5)  # ~2.5 R:R
                    side = "long"
                else:
                    stop = scan.price + atr_sl
                    target = scan.price - (atr_sl * 2.5)
                    side = "short"

                pos = self.account.open_trade(
                    symbol=scan.symbol,
                    side=side,
                    price=scan.price,
                    stop_loss=round(stop, 6),
                    target=round(target, 6),
                    score=scan.score,
                    reason=scan.reason,
                    avwap=scan.anchored_vwap,
                )
                if pos:
                    filled += 1
                    result["new_trades"].append({
                        "symbol": scan.symbol,
                        "side": side,
                        "price": scan.price,
                        "score": scan.score,
                        "reason": scan.reason,
                    })

            if filled > 0:
                result["action"] = f"opened_{filled}_trades"
            else:
                result["action"] = "monitoring"

        except Exception as e:
            result["error"] = str(e)
            log.exception(f"Bot cycle error: {e}")

        self._last_run_log = result
        return result

    @property
    def status(self) -> dict:
        session = get_current_session()
        positions_list = []
        for sym, pos in self.account.positions.items():
            price = self.market.get_price(sym)
            if pos.side == "long":
                unrealized = ((price or pos.entry_price) - pos.entry_price) / pos.entry_price * 100
            else:
                unrealized = (pos.entry_price - (price or pos.entry_price)) / pos.entry_price * 100

            positions_list.append({
                "symbol": sym,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "current_price": price,
                "unrealized_pct": round(unrealized, 2),
                "stop_loss": pos.stop_loss,
                "target": pos.target,
                "partial_closed": bool(pos.partial_closed),
                "is_runner": bool(pos.is_runner),
                "bars_held": pos.bars_held,
                "reason": pos.reason,
                "score": pos.score,
            })

        return {
            "account": self.account.stats,
            "session": str(session.value),
            "session_active": bool(is_session_active()),
            "positions": positions_list,
            "last_run": self._last_run_log,
            "universe": self.market._universe[:10],
        }
