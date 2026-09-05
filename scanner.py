"""
IQ Option Practice/Demo Signal Scanner
V42 replacement for V4/V4.1.

Signal-only engine:
- ACCOUNT_MODE = "PRACTICE"
- AUTO_TRADE = False
- Maximum 4 signals per calendar day (regular + OTC combined).
- Uses live IQ Option open-time status before candle analysis.
- Regular instruments are never treated as OTC instruments.
- OTC is considered only when the regular instrument is unavailable and the OTC
  symbol is explicitly reported as open by IQ Option.
- Core strategy is shared by regular and OTC instruments.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import requests

from iqoptionapi.stable_api import IQ_Option


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACCOUNT_MODE = "PRACTICE"
AUTO_TRADE = False

MAX_DAILY_SIGNALS = 4
STATE_FILE = Path(os.getenv("STATE_FILE", "v42_state.json"))
NIGERIA_TZ = ZoneInfo("Africa/Lagos")

IQ_EMAIL = os.getenv("IQ_EMAIL", "").strip()
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "").strip()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

# Regular instruments. OTC symbols are discovered from IQ Option's open-time
# catalogue instead of being assumed to be available.
REGULAR_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "CADJPY",
]

TIMEFRAMES = {
    "1M": 60,
    "5M": 300,
    "15M": 900,
}

CANDLE_COUNT = 220
MIN_CANDLES = 130

# Signal selectivity. A score is built from independent confirmations.
MIN_SCORE = 68
MIN_CONFIDENCE = 0.70
MIN_ATR_RATIO = 0.00015
MAX_ATR_RATIO = 0.012
MAX_SIGNAL_AGE_SECONDS = 180

# Avoid repeated alerts for essentially the same setup during one run.
SIGNAL_COOLDOWN_SECONDS = 20 * 60

# Small delay between pair scans to reduce pressure on the API.
PAIR_SCAN_DELAY_SECONDS = 0.35

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("v42_scanner")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    timestamp: int
    open: float
    close: float
    high: float
    low: float
    volume: float = 0.0


@dataclass
class Signal:
    pair: str
    market_type: str
    direction: str
    expiry_minutes: int
    score: int
    confidence: float
    entry_reference: float
    rsi: float
    bias_5m: str
    bias_15m: str
    reasons: List[str]
    generated_at: str


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def nigeria_today() -> str:
    return datetime.now(NIGERIA_TZ).date().isoformat()


def load_state() -> Dict[str, Any]:
    default = {
        "date": nigeria_today(),
        "signal_count": 0,
        "signals": [],
        "last_signal_epoch": 0.0,
    }

    try:
        if not STATE_FILE.exists():
            return default

        raw = STATE_FILE.read_text(encoding="utf-8")
        state = json.loads(raw)

        if not isinstance(state, dict):
            return default

        if state.get("date") != nigeria_today():
            state = default

        state.setdefault("signals", [])
        state.setdefault("last_signal_epoch", 0.0)

        try:
            state["signal_count"] = int(state.get("signal_count", 0))
        except (TypeError, ValueError):
            state["signal_count"] = 0

        state["signal_count"] = max(
            0, min(MAX_DAILY_SIGNALS, state["signal_count"])
        )
        return state
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read state file: %s", exc)
        return default


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(STATE_FILE)


def daily_limit_reached(state: Dict[str, Any]) -> bool:
    return int(state.get("signal_count", 0)) >= MAX_DAILY_SIGNALS


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def telegram_send(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        LOGGER.error("BOT_TOKEN and CHAT_ID must be set as environment variables.")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok", False):
            LOGGER.error("Telegram rejected message: %s", data)
            return False
        return True
    except requests.RequestException as exc:
        LOGGER.error("Telegram send failed: %s", exc)
        return False


def format_signal(signal: Signal, count_after_send: int) -> str:
    market_label = "OTC" if signal.market_type == "OTC" else "REGULAR"
    direction_emoji = "🟢 CALL" if signal.direction == "CALL" else "🔴 PUT"
    reason_text = "\n".join(f"• {reason}" for reason in signal.reasons)

    return (
        "📡 IQ OPTION PRACTICE SIGNAL\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"Pair: {signal.pair}\n"
        f"Market: {market_label}\n"
        f"Direction: {direction_emoji}\n"
        f"Adaptive expiry: {signal.expiry_minutes} min\n"
        f"Score: {signal.score}/100\n"
        f"Model confidence: {signal.confidence * 100:.1f}%\n"
        f"Entry reference: {signal.entry_reference:.6f}\n"
        f"RSI: {signal.rsi:.1f}\n"
        f"5M bias: {signal.bias_5m}\n"
        f"15M bias: {signal.bias_15m}\n"
        "\nReasons:\n"
        f"{reason_text}\n"
        "\n⚠️ PRACTICE/DEMO ONLY\n"
        "🤖 Auto-trading: OFF\n"
        f"Daily signals: {count_after_send}/{MAX_DAILY_SIGNALS} maximum\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Signal only — no automatic order is placed."
    )


# ---------------------------------------------------------------------------
# IQ Option connection and market status
# ---------------------------------------------------------------------------

def connect_iq() -> IQ_Option:
    if not IQ_EMAIL or not IQ_PASSWORD:
        raise RuntimeError(
            "IQ_EMAIL and IQ_PASSWORD must be supplied as environment variables."
        )

    iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
    connected, reason = iq.connect()
    if not connected:
        raise RuntimeError(f"IQ Option connection failed: {reason}")

    # The scanner is explicitly practice/demo only.
    iq.change_balance(ACCOUNT_MODE)
    LOGGER.info("Connected to IQ Option in %s mode.", ACCOUNT_MODE)

    if AUTO_TRADE:
        raise RuntimeError("AUTO_TRADE must remain False for this signal-only scanner.")

    return iq


def _normalise_open_time_catalog(raw: Any) -> Dict[str, Dict[str, Any]]:
    """
    IQ Option's open-time payload has varied slightly across API versions.
    This function keeps the status check conservative: if a symbol cannot be
    positively identified as open, it is treated as unavailable.
    """
    if not isinstance(raw, dict):
        return {}

    result: Dict[str, Dict[str, Any]] = {}

    for market_group, group_data in raw.items():
        if not isinstance(group_data, dict):
            continue

        for symbol, details in group_data.items():
            if isinstance(details, dict):
                item = dict(details)
                item["_group"] = market_group
                result[str(symbol).upper()] = item

    return result


def _details_confirm_open(details: Optional[Dict[str, Any]]) -> bool:
    if not details:
        return False

    # IQ Option commonly exposes "open". Other flags are accepted only when
    # explicitly true. No candle timestamp is ever used as an open-market test.
    if details.get("open") is True:
        return True
    if details.get("is_open") is True:
        return True
    if details.get("enabled") is True and details.get("open") is not False:
        return True

    return False


def market_status(iq: IQ_Option, regular_pair: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Lightweight market check.

    We do NOT use get_all_open_time() because that endpoint can hang.
    Instead, we test whether recent candles can be retrieved.
    """
    regular = regular_pair.upper()

    candidates = [
        (regular, "REGULAR"),
        (f"{regular}-OTC", "OTC"),
        (f"{regular}OTC", "OTC"),
        (f"{regular}.OTC", "OTC"),
        (f"{regular}_OTC", "OTC"),
    ]

    end_time = int(time.time())

    for symbol, market_type in candidates:
        try:
            candles = iq.get_candles(symbol, 60, 3, end_time)

            if candles and len(candles) >= 2:
                return symbol, market_type

        except Exception as exc:
            LOGGER.debug(
                "Market check failed for %s: %s",
                symbol,
                exc,
            )

    return None, None

def get_candles(iq: IQ_Option, symbol: str, timeframe: str, count: int) -> List[Candle]:
    seconds = TIMEFRAMES[timeframe]
    end = int(time.time())

    raw = iq.get_candles(symbol, seconds, count, end)
    if not raw:
        return []

    candles: List[Candle] = []
    for item in raw:
        try:
            candles.append(
                Candle(
                    timestamp=int(item["from"]),
                    open=float(item["open"]),
                    close=float(item["close"]),
                    high=float(item["max"]),
                    low=float(item["min"]),
                    volume=float(item.get("volume", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    candles.sort(key=lambda c: c.timestamp)
    return candles


def candles_are_fresh(candles: Sequence[Candle], timeframe_seconds: int) -> bool:
    """
    Freshness protects analysis quality only. It is deliberately NOT used to
    decide whether the market is open; open/closed status comes from
    get_all_open_time().
    """
    if not candles:
        return False

    age = int(time.time()) - int(candles[-1].timestamp)
    return age <= max(timeframe_seconds * 2, 180)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def ema(values: Sequence[float], period: int) -> List[float]:
    if len(values) < period:
        return []

    multiplier = 2.0 / (period + 1.0)
    result = [statistics.fmean(values[:period])]

    for value in values[period:]:
        result.append((value - result[-1]) * multiplier + result[-1])

    return result


def ema_last(values: Sequence[float], period: int) -> Optional[float]:
    values = list(values)
    if len(values) < period:
        return None
    return ema(values, period)[-1]


def rsi(values: Sequence[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = statistics.fmean(gains)
    avg_loss = statistics.fmean(losses)

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(values) < 35:
        return None, None, None

    fast = ema(values, 12)
    slow = ema(values, 26)
    if not fast or not slow:
        return None, None, None

    # Align fast EMA with slow EMA.
    offset = len(fast) - len(slow)
    macd_line = [fast[offset + i] - slow[i] for i in range(len(slow))]

    if len(macd_line) < 9:
        return None, None, None

    signal_line = ema(macd_line, 9)
    if not signal_line:
        return None, None, None

    current = macd_line[-1]
    signal = signal_line[-1]
    return current, signal, current - signal


def atr(candles: Sequence[Candle], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None

    true_ranges: List[float] = []
    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )

    if len(true_ranges) < period:
        return None

    return statistics.fmean(true_ranges[-period:])


def candle_momentum(candles: Sequence[Candle]) -> float:
    if len(candles) < 4:
        return 0.0

    sample = candles[-3:]
    weighted = 0.0
    total_weight = 0.0

    for idx, candle in enumerate(sample, start=1):
        body = candle.close - candle.open
        rng = max(candle.high - candle.low, 1e-12)
        weighted += (body / rng) * idx
        total_weight += idx

    return weighted / total_weight


def support_resistance(candles: Sequence[Candle], lookback: int = 40) -> Tuple[float, float]:
    sample = candles[-lookback:]
    support = min(c.low for c in sample)
    resistance = max(c.high for c in sample)
    return support, resistance


def percent_distance(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def bias_from_trend(
    closes: Sequence[float],
    fast_period: int = 20,
    mid_period: int = 50,
    slow_period: int = 100,
) -> str:
    e_fast = ema_last(closes, fast_period)
    e_mid = ema_last(closes, mid_period)
    e_slow = ema_last(closes, slow_period)

    if None in (e_fast, e_mid, e_slow):
        return "NEUTRAL"

    if e_fast > e_mid > e_slow:
        return "BULLISH"
    if e_fast < e_mid < e_slow:
        return "BEARISH"
    return "NEUTRAL"


def entry_structure(candles: Sequence[Candle]) -> Tuple[str, List[str]]:
    closes = [c.close for c in candles]
    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)

    if not e9 or not e21 or not e50:
        return "NEUTRAL", []

    # Use the latest aligned EMA values.
    latest9 = e9[-1]
    latest21 = e21[-1]
    latest50 = e50[-1]

    previous_closes = closes[-4:]
    pullback = False
    if len(previous_closes) >= 3:
        recent_low = min(candles[-3].low, candles[-2].low)
        recent_high = max(candles[-3].high, candles[-2].high)
        pullback = (
            recent_low <= latest21 or recent_low <= latest9
            or recent_high >= latest21 or recent_high >= latest9
        )

    reasons: List[str] = []

    if latest9 > latest21 > latest50:
        if pullback:
            reasons.append("EMA 9/21/50 bullish entry structure with pullback.")
        else:
            reasons.append("EMA 9/21/50 bullish alignment.")
        return "BULLISH", reasons

    if latest9 < latest21 < latest50:
        if pullback:
            reasons.append("EMA 9/21/50 bearish entry structure with pullback.")
        else:
            reasons.append("EMA 9/21/50 bearish alignment.")
        return "BEARISH", reasons

    return "NEUTRAL", reasons


def two_candle_confirmation(candles: Sequence[Candle], direction: str) -> bool:
    if len(candles) < 3:
        return False

    a = candles[-2]
    b = candles[-1]

    a_body = abs(a.close - a.open)
    b_body = abs(b.close - b.open)
    a_range = max(a.high - a.low, 1e-12)
    b_range = max(b.high - b.low, 1e-12)

    if direction == "CALL":
        return (
            a.close > a.open
            and b.close > b.open
            and b.close > a.close
            and (b_body / b_range) >= 0.50
            and (a_body / a_range) >= 0.35
        )

    return (
        a.close < a.open
        and b.close < b.open
        and b.close < a.close
        and (b_body / b_range) >= 0.50
        and (a_body / a_range) >= 0.35
    )


def momentum_confirmation(candles: Sequence[Candle], direction: str) -> bool:
    momentum = candle_momentum(candles)
    return momentum >= 0.22 if direction == "CALL" else momentum <= -0.22


def macd_confirmation(values: Sequence[float], direction: str) -> bool:
    current, signal, histogram = macd(values)
    if current is None or signal is None or histogram is None:
        return False

    if direction == "CALL":
        return current > signal and histogram > 0
    return current < signal and histogram < 0


def near_support_resistance(
    candles: Sequence[Candle],
    direction: str,
    atr_value: float,
) -> Tuple[bool, str]:
    support, resistance = support_resistance(candles)
    price = candles[-1].close
    proximity = max(atr_value * 0.90, price * 0.0005)

    if direction == "CALL":
        if abs(price - support) <= proximity:
            return True, "Price is reacting near established support."
        if price > resistance:
            return True, "Price has broken above the recent resistance zone."
    else:
        if abs(price - resistance) <= proximity:
            return True, "Price is reacting near established resistance."
        if price < support:
            return True, "Price has broken below the recent support zone."

    return False, ""


def volatility_ok(candles: Sequence[Candle], atr_value: float) -> bool:
    price = candles[-1].close
    ratio = atr_value / max(price, 1e-12)
    return MIN_ATR_RATIO <= ratio <= MAX_ATR_RATIO


def choose_expiry(
    candles_1m: Sequence[Candle],
    atr_value: float,
    score: int,
    direction: str,
) -> int:
    """
    Adaptive expiry:
      - 3m for strong, immediate momentum with healthy volatility.
      - 5m for the normal high-quality continuation.
      - 10m for slower but strongly aligned structure.
    """
    price = candles_1m[-1].close
    atr_ratio = atr_value / max(price, 1e-12)
    momentum = abs(candle_momentum(candles_1m))

    if score >= 90 and momentum >= 0.48 and atr_ratio >= MIN_ATR_RATIO * 1.4:
        return 3

    if score >= 84 and momentum >= 0.30:
        return 5

    return 10


def confidence_from_score(score: int, confirmations: int, conflicts: int) -> float:
    raw = 0.50 + (score / 100.0) * 0.40
    raw += min(confirmations, 4) * 0.025
    raw -= min(conflicts, 3) * 0.035
    return max(0.0, min(0.99, raw))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_symbol(
    iq: IQ_Option,
    symbol: str,
    market_type: str,
) -> Optional[Signal]:
    """
    Shared core strategy for both regular and OTC.

    1M = entry structure and trigger.
    5M = confirmation/bias.
    15M = higher-timeframe confirmation/bias.
    """
    candles_1m = get_candles(iq, symbol, "1M", CANDLE_COUNT)
    candles_5m = get_candles(iq, symbol, "5M", CANDLE_COUNT)
    candles_15m = get_candles(iq, symbol, "15M", CANDLE_COUNT)

    if any(
        len(candles) < MIN_CANDLES
        for candles in (candles_1m, candles_5m, candles_15m)
    ):
        LOGGER.debug("%s: insufficient candles.", symbol)
        return None

    if not candles_are_fresh(candles_1m, TIMEFRAMES["1M"]):
        LOGGER.debug("%s: 1M candles are stale.", symbol)
        return None
    if not candles_are_fresh(candles_5m, TIMEFRAMES["5M"]):
        LOGGER.debug("%s: 5M candles are stale.", symbol)
        return None
    if not candles_are_fresh(candles_15m, TIMEFRAMES["15M"]):
        LOGGER.debug("%s: 15M candles are stale.", symbol)
        return None

    closes_1m = [c.close for c in candles_1m]
    closes_5m = [c.close for c in candles_5m]
    closes_15m = [c.close for c in candles_15m]

    bias_5m = bias_from_trend(closes_5m, 20, 50, 100)
    bias_15m = bias_from_trend(closes_15m, 20, 50, 100)

    entry_bias, entry_reasons = entry_structure(candles_1m)
    if entry_bias == "NEUTRAL":
        return None

    direction = "CALL" if entry_bias == "BULLISH" else "PUT"

    # Higher-timeframe trend must agree; neutral is rejected rather than guessed.
    if direction == "CALL":
        if bias_5m != "BULLISH" or bias_15m != "BULLISH":
            return None
    else:
        if bias_5m != "BEARISH" or bias_15m != "BEARISH":
            return None

    rsi_value = rsi(closes_1m, 14)
    atr_value = atr(candles_1m, 14)

    if rsi_value is None or atr_value is None:
        return None

    if not volatility_ok(candles_1m, atr_value):
        return None

    # Avoid chasing deeply exhausted RSI.
    if direction == "CALL" and not (50.0 <= rsi_value <= 72.0):
        return None
    if direction == "PUT" and not (28.0 <= rsi_value <= 50.0):
        return None

    score = 0
    confirmations = 0
    conflicts = 0
    reasons = list(entry_reasons)

    # Higher timeframe trend: strongest structural component.
    score += 18
    confirmations += 1
    reasons.append("5M EMA 20/50/100 trend agrees with the entry.")
    score += 18
    confirmations += 1
    reasons.append("15M EMA 20/50/100 trend agrees with the entry.")

    # RSI.
    if direction == "CALL":
        if 54 <= rsi_value <= 67:
            score += 10
            confirmations += 1
            reasons.append("RSI is bullish without being deeply overbought.")
        elif 50 <= rsi_value < 54:
            score += 5
        else:
            conflicts += 1
    else:
        if 33 <= rsi_value <= 46:
            score += 10
            confirmations += 1
            reasons.append("RSI is bearish without being deeply oversold.")
        elif 46 < rsi_value <= 50:
            score += 5
        else:
            conflicts += 1

    # MACD.
    if macd_confirmation(closes_1m, direction):
        score += 12
        confirmations += 1
        reasons.append("MACD line/histogram confirms directional momentum.")
    else:
        conflicts += 1

    # ATR.
    score += 8
    confirmations += 1
    reasons.append("ATR volatility is inside the usable range.")

    # Candle momentum.
    if momentum_confirmation(candles_1m, direction):
        score += 10
        confirmations += 1
        reasons.append("Recent candle momentum confirms the direction.")
    else:
        conflicts += 1

    # Two-candle confirmation.
    if two_candle_confirmation(candles_1m, direction):
        score += 14
        confirmations += 1
        reasons.append("Two-candle continuation confirmation is present.")
    else:
        conflicts += 1

    # Support/resistance.
    sr_ok, sr_reason = near_support_resistance(candles_1m, direction, atr_value)
    if sr_ok:
        score += 10
        confirmations += 1
        reasons.append(sr_reason)
    else:
        score += 2

    # 5M/15M candle momentum must not contradict the structure.
    momentum_5m = candle_momentum(candles_5m)
    momentum_15m = candle_momentum(candles_15m)

    if direction == "CALL":
        if momentum_5m > -0.15 and momentum_15m > -0.15:
            score += 5
            confirmations += 1
            reasons.append("5M and 15M candle momentum are not materially bearish.")
        else:
            conflicts += 1
    else:
        if momentum_5m < 0.15 and momentum_15m < 0.15:
            score += 5
            confirmations += 1
            reasons.append("5M and 15M candle momentum are not materially bullish.")
        else:
            conflicts += 1

    score = max(0, min(100, int(round(score))))
    confidence = confidence_from_score(score, confirmations, conflicts)

    if score < MIN_SCORE or confidence < MIN_CONFIDENCE:
        return None

    # Do not issue a signal if the last candle is abnormally old despite the
    # earlier freshness checks.
    age = int(time.time()) - candles_1m[-1].timestamp
    if age > MAX_SIGNAL_AGE_SECONDS:
        return None

    expiry = choose_expiry(candles_1m, atr_value, score, direction)

    return Signal(
        pair=symbol,
        market_type=market_type,
        direction=direction,
        expiry_minutes=expiry,
        score=score,
        confidence=confidence,
        entry_reference=candles_1m[-1].close,
        rsi=rsi_value,
        bias_5m=bias_5m,
        bias_15m=bias_15m,
        reasons=reasons[:8],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def signal_key(signal: Signal) -> str:
    return f"{signal.pair}|{signal.market_type}|{signal.direction}"


def select_best_signal(
    iq: IQ_Option,
    state: Dict[str, Any],
) -> Optional[Signal]:
    if daily_limit_reached(state):
        LOGGER.info("Daily limit reached: %s/%s.", state["signal_count"], MAX_DAILY_SIGNALS)
        return None

    candidates: List[Signal] = []

    for regular_pair in REGULAR_PAIRS:
        if daily_limit_reached(state):
            break

        # IMPORTANT: status is checked BEFORE candle analysis.
        actual_symbol, market_type = market_status(iq, regular_pair)

        if actual_symbol is None or market_type is None:
            LOGGER.debug("%s unavailable; skipping.", regular_pair)
            continue

        LOGGER.debug(
            "%s -> %s (%s) is explicitly open according to IQ Option.",
            regular_pair,
            actual_symbol,
            market_type,
        )

        try:
            signal = analyze_symbol(iq, actual_symbol, market_type)
        except Exception as exc:
            LOGGER.warning("Analysis failed for %s: %s", actual_symbol, exc)
            signal = None

        if signal is not None:
            candidates.append(signal)

        time.sleep(PAIR_SCAN_DELAY_SECONDS)

    if not candidates:
        return None

    # Strongest qualifying setup only. No signal is manufactured to fill the
    # daily allowance.
    candidates.sort(
        key=lambda s: (
            s.score,
            s.confidence,
            1 if s.market_type == "REGULAR" else 0,
        ),
        reverse=True,
    )

    best = candidates[0]

    # Avoid immediate duplicate signal spam if the state already records one.
    now = time.time()
    last_signal_epoch = float(state.get("last_signal_epoch", 0.0))
    if now - last_signal_epoch < SIGNAL_COOLDOWN_SECONDS:
        LOGGER.info("Signal cooldown active; no new signal sent.")
        return None

    return best


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def validate_environment() -> None:
    missing = [
        name
        for name, value in {
            "IQ_EMAIL": IQ_EMAIL,
            "IQ_PASSWORD": IQ_PASSWORD,
            "BOT_TOKEN": BOT_TOKEN,
            "CHAT_ID": CHAT_ID,
        }.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    if ACCOUNT_MODE != "PRACTICE":
        raise RuntimeError("ACCOUNT_MODE must be exactly 'PRACTICE'.")

    if AUTO_TRADE:
        raise RuntimeError("AUTO_TRADE must be False.")


def record_sent_signal(state: Dict[str, Any], signal: Signal) -> None:
    new_count = int(state.get("signal_count", 0)) + 1
    if new_count > MAX_DAILY_SIGNALS:
        raise RuntimeError("Safety check blocked a fifth daily signal.")

    state["date"] = nigeria_today()
    state["signal_count"] = new_count
    state["last_signal_epoch"] = time.time()
    state.setdefault("signals", []).append(
        {
            **asdict(signal),
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    # Keep state compact.
    state["signals"] = state["signals"][-20:]
    save_state(state)


def main() -> None:
    validate_environment()

    state = load_state()

    if daily_limit_reached(state):
        LOGGER.info(
            "No scan required: daily limit is already %s/%s.",
            state["signal_count"],
            MAX_DAILY_SIGNALS,
        )
        return

    iq: Optional[IQ_Option] = None

    try:
        iq = connect_iq()
        telegram_send("✅ TEST: IQ Option scanner connected successfully.")

        # One run produces at most one new signal. If scheduled frequently,
        # every run re-evaluates the strongest currently qualifying setup.
        signal = select_best_signal(iq, state)

        if signal is None:
            LOGGER.info("No qualifying high-confidence setup found.")
            return

        # Final guard immediately before Telegram send.
        state = load_state()
        if daily_limit_reached(state):
            LOGGER.warning("Daily limit reached during final safety check.")
            return

        message = format_signal(
            signal,
            int(state.get("signal_count", 0)) + 1,
        )

        if telegram_send(message):
            record_sent_signal(state, signal)
            LOGGER.info(
                "Signal sent: %s %s %s | %s/%s.",
                signal.pair,
                signal.market_type,
                signal.direction,
                state["signal_count"],
                MAX_DAILY_SIGNALS,
            )
        else:
            LOGGER.error("Signal was not recorded because Telegram send failed.")

    except Exception as exc:
        LOGGER.exception("Scanner failed: %s", exc)
        raise
    finally:
        if iq is not None:
            try:
                iq.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
