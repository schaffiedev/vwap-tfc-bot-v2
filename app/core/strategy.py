"""
VWAP TFC v2 — Crypto-native Anchored VWAP Strategy
====================================================
Key changes from v1:
- Anchored VWAP (on swing high/low) instead of daily reset
- EMA stack trend filter (8 > 21 > 50 = bull)
- Session windows: US/EU/Asia opens only
- Multi-coin: scans top 25 by volume
- Exit system from Sniper V4: partial +3%, trail, runner +7%
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from datetime import datetime, timezone


class Trend(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SessionWindow(Enum):
    US_OPEN = "us_open"
    EU_OPEN = "eu_open"
    ASIA_OPEN = "asia_open"
    CLOSED = "closed"


@dataclass
class ScanResult:
    symbol: str
    score: float
    trend: Trend
    reason: str
    price: float
    anchored_vwap: float
    ema8: float
    ema21: float
    ema50: float
    volume_ratio: float
    rsi: float
    atr_pct: float


# ═══════════════════════════════════════════════
# SESSION DETECTION
# ═══════════════════════════════════════════════

def get_current_session(utc_now: Optional[datetime] = None) -> SessionWindow:
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)
    t = utc_now.hour * 60 + utc_now.minute
    if 13 * 60 + 30 <= t <= 16 * 60:
        return SessionWindow.US_OPEN
    if 7 * 60 <= t <= 10 * 60:
        return SessionWindow.EU_OPEN
    if 0 <= t <= 3 * 60:
        return SessionWindow.ASIA_OPEN
    return SessionWindow.CLOSED


def is_session_active(utc_now: Optional[datetime] = None) -> bool:
    return get_current_session(utc_now) != SessionWindow.CLOSED


# ═══════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def find_swing_anchor(df: pd.DataFrame, lookback: int = 50) -> Tuple[float, int]:
    """Find the most recent significant swing point to anchor VWAP."""
    if len(df) < lookback:
        lookback = len(df)
    recent = df.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values

    swing_points = []
    for i in range(2, len(highs) - 2):
        if highs[i] >= max(highs[i-1], highs[i-2], highs[i+1], highs[i+2]):
            swing_points.append((recent.index[i], highs[i], "high"))
        if lows[i] <= min(lows[i-1], lows[i-2], lows[i+1], lows[i+2]):
            swing_points.append((recent.index[i], lows[i], "low"))

    if not swing_points:
        return float(lows.min()), int(recent.index[np.argmin(lows)])
    last = swing_points[-1]
    return float(last[1]), int(last[0])


def compute_anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.Series:
    """VWAP anchored from a specific index."""
    mask = df.index >= anchor_idx
    subset = df[mask].copy()
    tp = (subset["high"] + subset["low"] + subset["close"]) / 3
    vol = subset["volume"].replace(0, 1).fillna(1)
    cum_tp_vol = (tp * vol).cumsum()
    cum_vol = vol.cumsum()
    avwap = cum_tp_vol / cum_vol
    result = pd.Series(np.nan, index=df.index)
    result[mask] = avwap
    return result


# ═══════════════════════════════════════════════
# TREND DETECTION
# ═══════════════════════════════════════════════

def detect_trend(df: pd.DataFrame) -> Trend:
    """EMA stack: 8 > 21 > 50 = bull, 8 < 21 < 50 = bear."""
    if len(df) < 55:
        return Trend.NEUTRAL
    ema8 = float(compute_ema(df["close"], 8).iloc[-1])
    ema21 = float(compute_ema(df["close"], 21).iloc[-1])
    ema50 = float(compute_ema(df["close"], 50).iloc[-1])
    if ema8 > ema21 > ema50:
        return Trend.BULLISH
    elif ema8 < ema21 < ema50:
        return Trend.BEARISH
    return Trend.NEUTRAL


# ═══════════════════════════════════════════════
# TFC SCORING (per coin)
# ═══════════════════════════════════════════════

def score_tfc_setup(df: pd.DataFrame) -> Optional[ScanResult]:
    """
    Score a single coin for TFC setup quality.
    Returns ScanResult if conditions met, None otherwise.
    """
    if len(df) < 55:
        return None

    close = df["close"]
    price = float(close.iloc[-1])
    high = float(df["high"].iloc[-1])
    low = float(df["low"].iloc[-1])
    prev_high = float(df["high"].iloc[-2])
    prev_low = float(df["low"].iloc[-2])

    # Indicators
    ema8 = compute_ema(close, 8)
    ema21 = compute_ema(close, 21)
    ema50 = compute_ema(close, 50)
    rsi = compute_rsi(close)
    atr = compute_atr(df)
    macd_line, macd_signal = compute_macd(close)

    ema8_v = float(ema8.iloc[-1])
    ema21_v = float(ema21.iloc[-1])
    ema50_v = float(ema50.iloc[-1])
    rsi_v = float(rsi.iloc[-1])
    atr_v = float(atr.iloc[-1])
    atr_pct = atr_v / price if price > 0 else 0
    macd_v = float(macd_line.iloc[-1])
    macd_s = float(macd_signal.iloc[-1])

    # Volume
    avg_vol = float(df["volume"].tail(20).mean())
    cur_vol = float(df["volume"].iloc[-1])
    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0

    # Hard reject: dead volume or extreme RSI
    if vol_ratio < 0.5:
        return None
    if rsi_v > 85 or rsi_v < 15:
        return None

    # Trend
    trend = detect_trend(df)
    if trend == Trend.NEUTRAL:
        return None

    # Anchored VWAP
    anchor_price, anchor_idx = find_swing_anchor(df)
    avwap = compute_anchored_vwap(df, anchor_idx)
    avwap_v = float(avwap.iloc[-1]) if not pd.isna(avwap.iloc[-1]) else price
    proximity = abs(price - avwap_v) / price if price > 0 else 1

    # ── TFC pattern detection (last 6 candles) ──
    tfc_found = False
    reason = ""

    if trend == Trend.BULLISH:
        if price <= avwap_v:
            return None
        # Was price below VWAP recently? (the "break")
        below_recently = any(
            float(close.iloc[i]) < float(avwap.iloc[i])
            for i in range(-6, -2) if not pd.isna(avwap.iloc[i])
        )
        # Did it retest? (came within 0.15%)
        retested = any(
            abs(float(close.iloc[i]) - float(avwap.iloc[i])) / max(float(close.iloc[i]), 0.001) < 0.0015
            for i in range(-4, -1) if not pd.isna(avwap.iloc[i])
        )
        # Continuation: breaks prev high
        cont = high > prev_high

        if below_recently and retested and cont:
            tfc_found = True
            reason = "TFC Long: AVWAP break → retest → continuation"
        elif proximity < 0.003 and cont and price > avwap_v:
            tfc_found = True
            reason = "AVWAP bounce long"

    elif trend == Trend.BEARISH:
        if price >= avwap_v:
            return None
        above_recently = any(
            float(close.iloc[i]) > float(avwap.iloc[i])
            for i in range(-6, -2) if not pd.isna(avwap.iloc[i])
        )
        retested = any(
            abs(float(close.iloc[i]) - float(avwap.iloc[i])) / max(float(close.iloc[i]), 0.001) < 0.0015
            for i in range(-4, -1) if not pd.isna(avwap.iloc[i])
        )
        cont = low < prev_low

        if above_recently and retested and cont:
            tfc_found = True
            reason = "TFC Short: AVWAP break → retest → continuation"
        elif proximity < 0.003 and cont and price < avwap_v:
            tfc_found = True
            reason = "AVWAP bounce short"

    if not tfc_found:
        return None

    # ── SCORING ──
    scores = []

    # Trend strength
    if trend == Trend.BULLISH:
        spread = (ema8_v - ema50_v) / price
    else:
        spread = (ema50_v - ema8_v) / price
    scores.append(min(1.0, spread / 0.02) if spread > 0 else 0)

    # RSI
    if 40 <= rsi_v <= 60:
        scores.append(0.95)
    elif 30 <= rsi_v <= 70:
        scores.append(0.70)
    else:
        scores.append(0.30)

    # Volume
    if vol_ratio >= 2.0:
        scores.append(1.0)
    elif vol_ratio >= 1.2:
        scores.append(0.80)
    else:
        scores.append(0.60)

    # MACD alignment
    if (trend == Trend.BULLISH and macd_v > macd_s) or \
       (trend == Trend.BEARISH and macd_v < macd_s):
        scores.append(0.90)
    else:
        scores.append(0.50)

    # VWAP proximity
    if proximity < 0.002:
        scores.append(1.0)
    elif proximity < 0.005:
        scores.append(0.80)
    else:
        scores.append(0.60)

    final = sum(scores) / len(scores)
    if final < 0.58:
        return None

    return ScanResult(
        symbol="",
        score=round(final, 3),
        trend=trend,
        reason=reason,
        price=price,
        anchored_vwap=round(avwap_v, 6),
        ema8=round(ema8_v, 6),
        ema21=round(ema21_v, 6),
        ema50=round(ema50_v, 6),
        volume_ratio=round(vol_ratio, 2),
        rsi=round(rsi_v, 1),
        atr_pct=round(atr_pct, 5),
    )
