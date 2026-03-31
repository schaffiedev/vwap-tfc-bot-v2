"""
Paper Trading Engine v2
========================
Multi-position with Sniper V4 exit system:
- Partial close 50% at +3%, move SL to breakeven
- Trail remaining at 1.8% below peak
- Full TP at +7% → keep 3% runner with 4% trail
- Stop loss: 2.5% (ATR-adaptive)
- Stale exit: 18 bars if < 0.5% move
- Daily loss limit: 5%
"""

import json
import os
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, List
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
TRADES_FILE = DATA_DIR / "trades_v2.json"
STATE_FILE = DATA_DIR / "state_v2.json"


@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    entry_time: str
    quantity: float
    stop_loss: float
    target: Optional[float] = None
    highest: float = 0.0
    lowest: float = 999999.0
    bars_held: int = 0
    partial_closed: bool = False
    partial_pnl: float = 0.0
    remaining_pct: float = 1.0
    trailing_stop: Optional[float] = None
    is_runner: bool = False
    reason: str = ""
    score: float = 0.0
    avwap_at_entry: float = 0.0


@dataclass
class Config:
    STARTING_BALANCE: float = 500.0
    MAX_POSITIONS: int = 5
    POSITION_PCT: float = 0.18        # 18% per slot (5×18%=90%)
    FEE: float = 0.0004               # Kraken maker

    # Entry
    MIN_SCORE: float = 0.58
    MAX_TRADES_PER_DAY: int = 8

    # Exits (Sniper V4)
    PARTIAL_TP_PCT: float = 0.03      # +3% → close 50%
    FULL_TP_PCT: float = 0.07         # +7% → close rest, keep runner
    STOP_LOSS_PCT: float = 0.025      # 2.5% SL
    TRAIL_DISTANCE: float = 0.018     # 1.8% trail after partial
    RUNNER_PCT: float = 0.03          # Keep 3% as runner
    RUNNER_TRAIL: float = 0.04        # 4% trail on runner

    # Stale
    STALE_BARS: int = 216             # 216 × 5min = 18 hours (matches Sniper V4)
    STALE_MIN_MOVE: float = 0.008     # 0.8% minimum move

    # Safety
    DAILY_LOSS_LIMIT: float = 0.05


class PaperAccount:
    def __init__(self):
        self.config = Config()
        self.balance: float = self.config.STARTING_BALANCE
        self.starting_balance: float = self.config.STARTING_BALANCE
        self.positions: Dict[str, Position] = {}
        self.trades: List[dict] = []
        self.equity_curve: List[dict] = []
        self.total_trades: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.total_pnl: float = 0.0
        self.peak_equity: float = self.config.STARTING_BALANCE
        self.max_drawdown: float = 0.0
        self.trades_today: int = 0
        self.today_date: str = ""
        self.daily_pnl: float = 0.0

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── Persistence ──────────────────────────────────

    def _load(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    d = json.load(f)
                self.balance = d.get("balance", self.config.STARTING_BALANCE)
                self.starting_balance = d.get("starting_balance", self.config.STARTING_BALANCE)
                self.total_trades = d.get("total_trades", 0)
                self.wins = d.get("wins", 0)
                self.losses = d.get("losses", 0)
                self.total_pnl = d.get("total_pnl", 0.0)
                self.peak_equity = d.get("peak_equity", self.config.STARTING_BALANCE)
                self.max_drawdown = d.get("max_drawdown", 0.0)
                self.equity_curve = d.get("equity_curve", [])
                self.trades_today = d.get("trades_today", 0)
                self.today_date = d.get("today_date", "")
                self.daily_pnl = d.get("daily_pnl", 0.0)
                # Restore open positions
                for sym, pos_data in d.get("positions", {}).items():
                    self.positions[sym] = Position(**pos_data)
            except Exception as e:
                log.error(f"Failed to load state: {e}")

        if TRADES_FILE.exists():
            try:
                with open(TRADES_FILE) as f:
                    self.trades = json.load(f)
            except Exception:
                pass

    def save(self):
        state = {
            "balance": self.balance,
            "starting_balance": self.starting_balance,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": self.total_pnl,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "equity_curve": self.equity_curve[-500:],
            "trades_today": self.trades_today,
            "today_date": self.today_date,
            "daily_pnl": self.daily_pnl,
            "positions": {sym: asdict(pos) for sym, pos in self.positions.items()},
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        with open(TRADES_FILE, "w") as f:
            json.dump(self.trades[-200:], f, indent=2, default=str)

    # ── Daily reset ──────────────────────────────────

    def _check_daily_reset(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.today_date != today:
            self.today_date = today
            self.trades_today = 0
            self.daily_pnl = 0.0

    # ── Slot management ──────────────────────────────

    @property
    def open_slots(self) -> int:
        return self.config.MAX_POSITIONS - len(self.positions)

    @property
    def can_trade(self) -> bool:
        self._check_daily_reset()
        if self.open_slots <= 0:
            return False
        if self.trades_today >= self.config.MAX_TRADES_PER_DAY:
            return False
        if self.daily_pnl <= -(self.balance * self.config.DAILY_LOSS_LIMIT):
            return False
        return True

    # ── Open trade ───────────────────────────────────

    def open_trade(self, symbol: str, side: str, price: float,
                   stop_loss: float, target: float, score: float,
                   reason: str, avwap: float) -> Optional[Position]:
        if symbol in self.positions:
            return None
        if not self.can_trade:
            return None

        # Position size
        alloc = self.balance * self.config.POSITION_PCT
        quantity = alloc / price
        if quantity <= 0:
            return None

        self.total_trades += 1
        self.trades_today += 1

        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            quantity=round(quantity, 8),
            stop_loss=stop_loss,
            target=target,
            highest=price,
            lowest=price,
            reason=reason,
            score=score,
            avwap_at_entry=avwap,
        )
        self.positions[symbol] = pos
        log.info(f"📥 OPEN {side.upper()} {symbol} @ ${price:.4f} | "
                 f"qty={quantity:.6f} | SL=${stop_loss:.4f} | TP=${target:.4f} | "
                 f"score={score:.3f} | [{len(self.positions)}/{self.config.MAX_POSITIONS} slots]")
        self.save()
        return pos

    # ── Update position (called every cycle) ─────────

    def update_position(self, symbol: str, current_price: float) -> Optional[dict]:
        """
        Update position with current price. Returns trade result dict if closed.
        Implements Sniper V4 exit cascade.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        pos.bars_held += 1

        # Update high/low tracking
        if current_price > pos.highest:
            pos.highest = current_price
        if current_price < pos.lowest:
            pos.lowest = current_price

        price = current_price

        if pos.side == "long":
            pnl_pct = (price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - price) / pos.entry_price

        # ── 1. STOP LOSS ────────────────────────────
        hit_stop = False
        if pos.side == "long" and price <= pos.stop_loss:
            hit_stop = True
        elif pos.side == "short" and price >= pos.stop_loss:
            hit_stop = True

        if hit_stop:
            return self._close_position(symbol, price, "STOP_LOSS")

        # ── 2. TRAILING STOP (after partial) ────────
        if pos.partial_closed and not pos.is_runner:
            if pos.side == "long":
                new_trail = pos.highest * (1 - self.config.TRAIL_DISTANCE)
                if pos.trailing_stop is None or new_trail > pos.trailing_stop:
                    pos.trailing_stop = new_trail
                if price <= pos.trailing_stop:
                    return self._close_position(symbol, price, "TRAILING_STOP")
            else:
                new_trail = pos.lowest * (1 + self.config.TRAIL_DISTANCE)
                if pos.trailing_stop is None or new_trail < pos.trailing_stop:
                    pos.trailing_stop = new_trail
                if price >= pos.trailing_stop:
                    return self._close_position(symbol, price, "TRAILING_STOP")

        # ── 3. RUNNER TRAILING ──────────────────────
        if pos.is_runner:
            if pos.side == "long":
                new_trail = pos.highest * (1 - self.config.RUNNER_TRAIL)
                if pos.trailing_stop is None or new_trail > pos.trailing_stop:
                    pos.trailing_stop = new_trail
                if price <= pos.trailing_stop:
                    return self._close_position(symbol, price, "RUNNER_TRAIL")
            else:
                new_trail = pos.lowest * (1 + self.config.RUNNER_TRAIL)
                if pos.trailing_stop is None or new_trail < pos.trailing_stop:
                    pos.trailing_stop = new_trail
                if price >= pos.trailing_stop:
                    return self._close_position(symbol, price, "RUNNER_TRAIL")

        # ── 4. PARTIAL CLOSE at +3% ────────────────
        if not pos.partial_closed and pnl_pct >= self.config.PARTIAL_TP_PCT:
            close_qty = pos.quantity * 0.50
            if pos.side == "long":
                partial_pnl = (price - pos.entry_price) * close_qty
            else:
                partial_pnl = (pos.entry_price - price) * close_qty
            partial_pnl -= price * close_qty * self.config.FEE

            pos.partial_closed = True
            pos.partial_pnl = partial_pnl
            pos.remaining_pct = 0.50
            # Move stop to breakeven + 0.2%
            if pos.side == "long":
                pos.stop_loss = pos.entry_price * 1.002
            else:
                pos.stop_loss = pos.entry_price * 0.998

            log.info(f"  ✂️ PARTIAL {symbol}: 50% closed at +{pnl_pct*100:.1f}% "
                     f"(${partial_pnl:+.2f}) → SL moved to breakeven")
            self.save()

        # ── 5. FULL TP at +7% → convert to runner ──
        if pos.partial_closed and not pos.is_runner and pnl_pct >= self.config.FULL_TP_PCT:
            # Close most, keep 3% as runner
            runner_qty = pos.quantity * pos.remaining_pct * self.config.RUNNER_PCT
            sell_qty = pos.quantity * pos.remaining_pct - runner_qty

            if pos.side == "long":
                sell_pnl = (price - pos.entry_price) * sell_qty
            else:
                sell_pnl = (pos.entry_price - price) * sell_qty
            sell_pnl -= price * sell_qty * self.config.FEE

            total_pnl = pos.partial_pnl + sell_pnl
            self.balance += total_pnl
            self.total_pnl += total_pnl
            self.daily_pnl += total_pnl
            self.wins += 1

            # Log the main trade as closed
            self.trades.append({
                "id": self.total_trades,
                "symbol": symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "entry_time": pos.entry_time,
                "exit_price": price,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "pnl": round(total_pnl, 2),
                "pnl_pct": round(pnl_pct * 100, 2),
                "reason": "FULL_TP (+runner)",
                "bars_held": pos.bars_held,
                "score": pos.score,
            })

            self._update_equity(price)

            if runner_qty * price > 1:  # Keep runner if > $1
                pos.is_runner = True
                pos.quantity = runner_qty
                pos.remaining_pct = 1.0
                pos.trailing_stop = None
                log.info(f"  🎯 FULL TP {symbol}: +{pnl_pct*100:.1f}% (${total_pnl:+.2f}) "
                         f"→ runner kept ({runner_qty:.6f})")
            else:
                del self.positions[symbol]
                log.info(f"  🎯 FULL TP {symbol}: +{pnl_pct*100:.1f}% (${total_pnl:+.2f})")

            self.save()
            return {"action": "full_tp", "pnl": total_pnl}

        # ── 6. STALE EXIT ──────────────────────────
        if pos.bars_held >= self.config.STALE_BARS and not pos.partial_closed:
            move = abs(pnl_pct)
            if move < self.config.STALE_MIN_MOVE:
                return self._close_position(symbol, price, "STALE")

        self.save()
        return None

    # ── Close position ───────────────────────────────

    def _close_position(self, symbol: str, price: float, reason: str) -> dict:
        pos = self.positions[symbol]

        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.quantity * pos.remaining_pct
        else:
            pnl = (pos.entry_price - price) * pos.quantity * pos.remaining_pct
        pnl -= price * pos.quantity * pos.remaining_pct * self.config.FEE
        pnl += pos.partial_pnl  # Add partial profits

        pnl_pct = pnl / (pos.entry_price * pos.quantity) * 100

        self.balance += pnl
        self.total_pnl += pnl
        self.daily_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        self._update_equity(price)

        emoji = "🟢" if pnl > 0 else "🔴"
        log.info(f"  {emoji} CLOSE {symbol}: {reason} @ ${price:.4f} | "
                 f"PnL=${pnl:+.2f} ({pnl_pct:+.1f}%) | "
                 f"Balance=${self.balance:.2f}")

        trade = {
            "id": self.total_trades,
            "symbol": symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "entry_time": pos.entry_time,
            "exit_price": price,
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
            "bars_held": pos.bars_held,
            "score": pos.score,
            "stop_loss": pos.stop_loss,
            "target": pos.target,
        }
        self.trades.append(trade)
        del self.positions[symbol]
        self.save()
        return trade

    def _update_equity(self, price: float):
        if self.balance > self.peak_equity:
            self.peak_equity = self.balance
        dd = (self.peak_equity - self.balance) / self.peak_equity if self.peak_equity > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd
        self.equity_curve.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "equity": round(self.balance, 2),
        })

    # ── Stats ────────────────────────────────────────

    @property
    def stats(self) -> dict:
        closed = self.wins + self.losses
        wr = (self.wins / closed * 100) if closed > 0 else 0
        return {
            "balance": round(self.balance, 2),
            "starting_balance": self.starting_balance,
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round((self.total_pnl / self.starting_balance) * 100, 2),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(wr, 1),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "peak_equity": round(self.peak_equity, 2),
            "open_positions": len(self.positions),
            "open_slots": self.open_slots,
            "trades_today": self.trades_today,
            "daily_pnl": round(self.daily_pnl, 2),
        }
