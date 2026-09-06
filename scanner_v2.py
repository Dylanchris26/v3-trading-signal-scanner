"""
IQ Option Practice/Demo Signal Scanner
Improved V2 practice/demo signal scanner.

- Practice/demo only
- Auto-trading OFF
- Maximum 4 signals/day
- Regular pairs first; OTC fallback
- 1M entry + 5M/15M trend confirmation
- Telegram alerts
- Detailed rejection diagnostics
- Uses candle "to" timestamp when available so freshness is based on the
  candle's close/update time rather than only its opening time.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import requests
from iqoptionapi.stable_api import IQ_Option
import iqoptionapi.constants as OP_code


# ============================================================
# CONFIGURATION
# ============================================================

ACCOUNT_MODE = "PRACTICE"
AUTO_TRADE = False

MAX_DAILY_SIGNALS = 4
STATE_FILE = Path(os.getenv("STATE_FILE", "v2_state.json"))

NIGERIA_TZ = ZoneInfo("Africa/Lagos")

IQ_EMAIL = os.getenv("IQ_EMAIL", "").strip()
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "").strip()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

REGULAR_PAIRS = [
    # Data-informed V2 universe: pairs that reached at least 57.5% in the
    # first 384-signal baseline test. Weak baseline pairs are excluded until
    # a fresh validation test proves they deserve to return.
    "AUDJPY",
    "AUDUSD",
    "CADJPY",
    "EURJPY",
    "GBPJPY",
    "GBPUSD",
    "USDJPY",
]

TIMEFRAMES = {
    "1M": 60,
    "5M": 300,
    "15M": 900,
}

CANDLE_COUNT = 220
MIN_CANDLES = 130

# Signal quality gates.
MIN_SCORE_CALL = 80
MIN_SCORE_PUT = 76
MIN_CONFIDENCE = 0.74

# ATR must be neither dead nor excessively volatile.
MIN_ATR_RATIO = 0.00015
MAX_ATR_RATIO = 0.012

# Latest candle may be up to 180 seconds old.
MAX_SIGNAL_AGE_SECONDS = 180

SIGNAL_COOLDOWN_SECONDS = 20 * 60
V2_EXPIRY_MINUTES = 10
MAX_EMA_DISTANCE_ATR = 1.15
MIN_TREND_STRENGTH_ATR = 0.08
PAIR_SCAN_DELAY_SECONDS = 0.35

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("v2_scanner")


# ============================================================
# DATA MODELS
# ============================================================

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


# ============================================================
# STATE
# ============================================================

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

        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))

        if not isinstance(state, dict):
            return default

        if state.get("date") != nigeria_today():
            return default

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


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        LOGGER.error("BOT_TOKEN and CHAT_ID must be set.")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            json={
                "chat_id": CHAT_ID,
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
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
    direction = "🟢 CALL" if signal.direction == "CALL" else "🔴 PUT"
    reasons = "\n".join(f"• {x}" for x in signal.reasons)

    return (
        "📡 IQ OPTION PRACTICE SIGNAL\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"Pair: {signal.pair}\n"
        f"Market: {signal.market_type}\n"
        f"Direction: {direction}\n"
        f"Expiry: {signal.expiry_minutes} min\n"
        f"Score: {signal.score}/100\n"
        f"Quality index: {signal.confidence * 100:.1f}%\n"
        f"Entry: {signal.entry_reference:.6f}\n"
        f"RSI: {signal.rsi:.1f}\n"
        f"5M bias: {signal.bias_5m}\n"
        f"15M bias: {signal.bias_15m}\n"
        "\nReasons:\n"
        f"{reasons}\n"
        "\n⚠️ PRACTICE/DEMO ONLY\n"
        "🤖 Auto-trading: OFF\n"
        f"Daily signals: {count_after_send}/{MAX_DAILY_SIGNALS}\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# IQ OPTION
# ============================================================

def connect_iq() -> IQ_Option:
    if not IQ_EMAIL or not IQ_PASSWORD:
        raise RuntimeError(
            "IQ_EMAIL and IQ_PASSWORD must be supplied as environment variables."
        )

    iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
    connected, reason = iq.connect()

    if not connected:
        raise RuntimeError(f"IQ Option connection failed: {reason}")

    iq.change_balance(ACCOUNT_MODE)

    # IMPORTANT: do NOT call update_ACTIVES_OPCODE() here.
    # That helper also requests crypto/forex/CFD instruments, and some
    # iqoptionapi builds now receive an "Invalid contract" response there.
    # We only need the binary/turbo initialization payload for OTC symbols.
    try:
        init_data = iq.get_all_init_v2()
        mapped = 0
        if isinstance(init_data, dict):
            for option_type in ("binary", "turbo"):
                section = init_data.get(option_type, {})
                actives = section.get("actives", {}) if isinstance(section, dict) else {}
                if not isinstance(actives, dict):
                    continue
                for active_id, active in actives.items():
                    if not isinstance(active, dict):
                        continue
                    raw_name = str(active.get("name", "")).strip()
                    if not raw_name:
                        continue
                    name = raw_name.split(".")[-1].upper()
                    try:
                        OP_code.ACTIVES[name] = int(active_id)
                        mapped += 1
                    except (TypeError, ValueError):
                        continue
        LOGGER.info("IQ Option binary/turbo symbols refreshed: %s symbols.", mapped)
    except Exception as exc:
        LOGGER.warning("Could not refresh IQ Option binary/turbo symbols: %s", exc)

    if AUTO_TRADE:
        raise RuntimeError("AUTO_TRADE must remain False.")

    LOGGER.info("Connected to IQ Option in %s mode.", ACCOUNT_MODE)

    return iq


def _iq_server_time(iq: IQ_Option) -> int:
    """Return IQ Option server time when available."""
    try:
        ts = getattr(getattr(iq, "timesync", None), "server_timestamp", None)
        if ts is not None:
            return int(float(ts))
    except (TypeError, ValueError):
        pass
    return int(time.time())


def _latest_candle_timestamp(raw: Any) -> Optional[int]:
    if not raw:
        return None
    try:
        item = raw[-1]
        value = item.get("to", item.get("from"))
        if value is None:
            return None
        ts = int(float(value))
        if ts > 10_000_000_000:
            ts //= 1000
        return ts
    except (IndexError, AttributeError, TypeError, ValueError):
        return None


_OTC_SYMBOL_CACHE: Dict[str, List[str]] = {}


def _discover_otc_symbols(iq: IQ_Option, regular_pair: str) -> List[str]:
    """Discover exact OTC symbols from IQ Option's binary/turbo init data."""
    regular = regular_pair.upper()
    if regular in _OTC_SYMBOL_CACHE:
        return list(_OTC_SYMBOL_CACHE[regular])

    found: List[str] = []
    try:
        data = iq.get_all_init_v2()
        if not isinstance(data, dict):
            return found

        for option_type in ("binary", "turbo"):
            section = data.get(option_type, {})
            actives = section.get("actives", {}) if isinstance(section, dict) else {}
            if not isinstance(actives, dict):
                continue

            for active_id, active in actives.items():
                if not isinstance(active, dict):
                    continue
                raw_name = str(active.get("name", "")).strip()
                if not raw_name:
                    continue
                name = raw_name.split(".")[-1].upper()

                # Keep the live IQ Option name -> active-id mapping in sync
                # without touching crypto/forex/CFD instrument endpoints.
                try:
                    OP_code.ACTIVES[name] = int(active_id)
                except (TypeError, ValueError):
                    pass

                if name == f"{regular}-OTC" and name not in found:
                    found.append(name)
    except Exception as exc:
        LOGGER.debug("Could not discover OTC assets for %s: %s", regular_pair, exc)

    _OTC_SYMBOL_CACHE[regular] = list(found)
    return found


def market_status(
    iq: IQ_Option,
    regular_pair: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Choose a live regular pair, otherwise the exact live OTC symbol."""
    regular = regular_pair.upper()
    server_now = _iq_server_time(iq)
    freshness_limit = max(TIMEFRAMES["1M"] * 2, MAX_SIGNAL_AGE_SECONDS)

    candidates: List[Tuple[str, str]] = [(regular, "REGULAR")]
    discovered = _discover_otc_symbols(iq, regular)
    candidates.extend((symbol, "OTC") for symbol in discovered)

    # Compatibility fallbacks for older API builds.
    for symbol in (
        f"{regular}-OTC",
        f"{regular}OTC",
        f"{regular}.OTC",
        f"{regular}_OTC",
    ):
        if not any(existing == symbol for existing, _ in candidates):
            candidates.append((symbol, "OTC"))

    for symbol, market_type in candidates:
        try:
            raw = iq.get_candles(symbol, 60, 3, server_now)
            latest = _latest_candle_timestamp(raw)
            if latest is None or not raw or len(raw) < 2:
                LOGGER.debug("MARKET CHECK %s (%s): no usable candles.", symbol, market_type)
                continue
            age = max(0, server_now - latest)
            if age <= freshness_limit:
                LOGGER.info(
                    "MARKET CHECK %s -> %s (%s): LIVE (age=%ss).",
                    regular_pair, symbol, market_type, age,
                )
                return symbol, market_type
            LOGGER.debug(
                "MARKET CHECK %s (%s): stale (age=%ss); trying next market.",
                symbol, market_type, age,
            )
        except Exception as exc:
            LOGGER.debug("Market check failed for %s: %s", symbol, exc)

    LOGGER.warning("MARKET CHECK %s: no LIVE regular or OTC market found.", regular_pair)
    return None, None


# ============================================================
# CANDLE DATA
# ============================================================

def get_candles(
    iq: IQ_Option,
    symbol: str,
    timeframe: str,
    count: int,
) -> List[Candle]:

    seconds = TIMEFRAMES[timeframe]
    end = _iq_server_time(iq)

    try:
        raw = iq.get_candles(symbol, seconds, count, end)
    except Exception as exc:
        LOGGER.debug(
            "%s %s candle request failed: %s",
            symbol,
            timeframe,
            exc,
        )
        return []

    if not raw:
        return []

    candles: List[Candle] = []

    for item in raw:
        try:
            # IQ Option candle data normally has both "from" and "to".
            # "to" is preferable for freshness because it represents the
            # candle's close/update time.
            raw_timestamp = item.get("to", item.get("from"))

            if raw_timestamp is None:
                continue

            timestamp = int(float(raw_timestamp))

            # Defensive support for millisecond timestamps.
            if timestamp > 10_000_000_000:
                timestamp //= 1000

            candles.append(
                Candle(
                    timestamp=timestamp,
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

    if candles:
        age = _iq_server_time(iq) - candles[-1].timestamp
        LOGGER.debug(
            "%s %s candles: count=%d latest_timestamp=%d age=%ds",
            symbol,
            timeframe,
            len(candles),
            candles[-1].timestamp,
            age,
        )

    return candles


def candles_are_fresh(
    candles: Sequence[Candle],
    timeframe_seconds: int,
) -> bool:

    if not candles:
        return False

    age = int(time.time()) - int(candles[-1].timestamp)

    # Never reject a future timestamp caused by feed timing.
    if age < 0:
        return True

    allowed = max(timeframe_seconds * 2, MAX_SIGNAL_AGE_SECONDS)

    return age <= allowed


# ============================================================
# INDICATORS
# ============================================================

def ema(values: Sequence[float], period: int) -> List[float]:
    if len(values) < period:
        return []

    multiplier = 2.0 / (period + 1.0)
    result = [statistics.fmean(values[:period])]

    for value in values[period:]:
        result.append(
            (value - result[-1]) * multiplier + result[-1]
        )

    return result


def ema_last(
    values: Sequence[float],
    period: int,
) -> Optional[float]:

    if len(values) < period:
        return None

    result = ema(values, period)
    return result[-1] if result else None


def rsi(
    values: Sequence[float],
    period: int = 14,
) -> Optional[float]:

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


def macd(
    values: Sequence[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:

    if len(values) < 35:
        return None, None, None

    fast = ema(values, 12)
    slow = ema(values, 26)

    if not fast or not slow:
        return None, None, None

    offset = len(fast) - len(slow)
    macd_line = [
        fast[offset + i] - slow[i]
        for i in range(len(slow))
    ]

    signal_line = ema(macd_line, 9)

    if not signal_line:
        return None, None, None

    current = macd_line[-1]
    signal = signal_line[-1]

    return current, signal, current - signal


def atr(
    candles: Sequence[Candle],
    period: int = 14,
) -> Optional[float]:

    if len(candles) < period + 1:
        return None

    ranges = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )

    if len(ranges) < period:
        return None

    return statistics.fmean(ranges[-period:])


def candle_momentum(candles: Sequence[Candle]) -> float:
    if len(candles) < 4:
        return 0.0

    sample = candles[-3:]
    weighted = 0.0
    total_weight = 0.0

    for weight, candle in enumerate(sample, start=1):
        body = candle.close - candle.open
        rng = max(candle.high - candle.low, 1e-12)

        weighted += (body / rng) * weight
        total_weight += weight

    return weighted / total_weight


# ============================================================
# STRATEGY
# ============================================================

def bias_from_trend(
    closes: Sequence[float],
    fast_period: int = 20,
    mid_period: int = 50,
    slow_period: int = 100,
) -> str:

    fast = ema_last(closes, fast_period)
    mid = ema_last(closes, mid_period)
    slow = ema_last(closes, slow_period)

    if None in (fast, mid, slow):
        return "NEUTRAL"

    if fast > mid > slow:
        return "BULLISH"

    if fast < mid < slow:
        return "BEARISH"

    return "NEUTRAL"


def entry_structure(
    candles: Sequence[Candle],
) -> Tuple[str, List[str]]:

    closes = [c.close for c in candles]

    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)

    if not e9 or not e21 or not e50:
        return "NEUTRAL", []

    latest9 = e9[-1]
    latest21 = e21[-1]
    latest50 = e50[-1]

    reasons: List[str] = []

    if latest9 > latest21 > latest50:
        reasons.append("1M EMA 9/21/50 is bullish.")
        return "BULLISH", reasons

    if latest9 < latest21 < latest50:
        reasons.append("1M EMA 9/21/50 is bearish.")
        return "BEARISH", reasons

    return "NEUTRAL", reasons


def two_candle_confirmation(
    candles: Sequence[Candle],
    direction: str,
) -> bool:

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


def macd_confirmation(
    values: Sequence[float],
    direction: str,
) -> bool:

    current, signal, histogram = macd(values)

    if current is None or signal is None or histogram is None:
        return False

    if direction == "CALL":
        return current > signal and histogram > 0

    return current < signal and histogram < 0


def support_resistance(
    candles: Sequence[Candle],
    lookback: int = 40,
) -> Tuple[float, float]:

    sample = candles[-lookback:]

    return (
        min(c.low for c in sample),
        max(c.high for c in sample),
    )


def near_support_resistance(
    candles: Sequence[Candle],
    direction: str,
    atr_value: float,
) -> Tuple[bool, str]:

    support, resistance = support_resistance(candles)
    price = candles[-1].close

    proximity = max(
        atr_value * 0.90,
        price * 0.0005,
    )

    if direction == "CALL":
        if abs(price - support) <= proximity:
            return True, "Price is reacting near support."

        if price > resistance:
            return True, "Price broke above resistance."

    else:
        if abs(price - resistance) <= proximity:
            return True, "Price is reacting near resistance."

        if price < support:
            return True, "Price broke below support."

    return False, ""


def volatility_ok(
    candles: Sequence[Candle],
    atr_value: float,
) -> bool:

    price = candles[-1].close
    ratio = atr_value / max(price, 1e-12)

    return MIN_ATR_RATIO <= ratio <= MAX_ATR_RATIO


def confidence_from_score(
    score: int,
    confirmations: int,
    conflicts: int,
) -> float:
    """Quality index only; NOT a probability of winning."""
    value = 0.50 + (score / 100.0) * 0.42
    value += min(confirmations, 5) * 0.018
    value -= min(conflicts, 2) * 0.025
    return max(0.0, min(0.99, value))


def choose_expiry(
    candles: Sequence[Candle],
    atr_value: float,
    score: int,
) -> int:
    # V2 uses the only expiry that cleared 50% in the baseline test.
    return V2_EXPIRY_MINUTES


def trend_strength_ok(
    candles: Sequence[Candle],
    direction: str,
) -> bool:
    closes = [c.close for c in candles]
    e20 = ema_last(closes, 20)
    e50 = ema_last(closes, 50)
    a = atr(candles, 14)
    if e20 is None or e50 is None or a is None or a <= 0:
        return False
    strength = abs(e20 - e50) / a
    if strength < MIN_TREND_STRENGTH_ATR:
        return False
    return e20 > e50 if direction == "CALL" else e20 < e50


def pullback_or_breakout_confirmation(
    candles: Sequence[Candle],
    direction: str,
    atr_value: float,
) -> Tuple[bool, str]:
    if len(candles) < 25:
        return False, ""
    closes = [c.close for c in candles]
    e9 = ema_last(closes, 9)
    e21 = ema_last(closes, 21)
    if e9 is None or e21 is None:
        return False, ""
    last = candles[-1]
    prev = candles[-2]
    distance = abs(last.close - e21) / max(atr_value, 1e-12)

    if direction == "CALL":
        if (prev.close <= e21 and last.close > e21 and last.close > last.open
                and distance <= MAX_EMA_DISTANCE_ATR):
            return True, "1M bullish EMA21 reclaim after pullback."
        if (last.close > e9 and last.close > last.open
                and prev.close <= e9 and distance <= MAX_EMA_DISTANCE_ATR):
            return True, "1M bullish EMA9 reclaim after pullback."
        return False, ""

    if (prev.close >= e21 and last.close < e21 and last.close < last.open
            and distance <= MAX_EMA_DISTANCE_ATR):
        return True, "1M bearish EMA21 reclaim after pullback."
    if (last.close < e9 and last.close < last.open
            and prev.close >= e9 and distance <= MAX_EMA_DISTANCE_ATR):
        return True, "1M bearish EMA9 reclaim after pullback."
    return False, ""


def room_to_opposite_level(
    candles: Sequence[Candle],
    direction: str,
    atr_value: float,
) -> bool:
    support, resistance = support_resistance(candles, lookback=60)
    price = candles[-1].close
    room = 0.55 * atr_value
    if direction == "CALL":
        return resistance - price > room or price > resistance
    return price - support > room or price < support


def analyze_setup(
    candles_1m: Sequence[Candle],
    candles_5m: Sequence[Candle],
    candles_15m: Sequence[Candle],
) -> Optional[Dict[str, Any]]:
    """Pure V2 strategy evaluation. No network/live freshness checks."""
    if any(len(c) < MIN_CANDLES for c in (candles_1m, candles_5m, candles_15m)):
        return None

    closes_1m = [c.close for c in candles_1m]
    closes_5m = [c.close for c in candles_5m]
    closes_15m = [c.close for c in candles_15m]

    bias_5m = bias_from_trend(closes_5m)
    bias_15m = bias_from_trend(closes_15m)
    entry_bias, entry_reasons = entry_structure(candles_1m)
    if entry_bias == "NEUTRAL":
        return None

    direction = "CALL" if entry_bias == "BULLISH" else "PUT"
    if direction == "CALL" and not (bias_5m == "BULLISH" and bias_15m == "BULLISH"):
        return None
    if direction == "PUT" and not (bias_5m == "BEARISH" and bias_15m == "BEARISH"):
        return None

    rsi_value = rsi(closes_1m, 14)
    atr_value = atr(candles_1m, 14)
    if rsi_value is None or atr_value is None or not volatility_ok(candles_1m, atr_value):
        return None

    # V2 narrows RSI to avoid late/overextended entries.
    if direction == "CALL" and not (54.0 <= rsi_value <= 66.0):
        return None
    if direction == "PUT" and not (34.0 <= rsi_value <= 47.0):
        return None

    # Hard confirmations: trend strength + MACD + pullback/reclaim.
    if not trend_strength_ok(candles_5m, direction):
        return None
    if not trend_strength_ok(candles_15m, direction):
        return None
    if not macd_confirmation(closes_1m, direction):
        return None
    pullback_ok, pullback_reason = pullback_or_breakout_confirmation(
        candles_1m, direction, atr_value
    )
    if not pullback_ok:
        return None
    if not room_to_opposite_level(candles_1m, direction, atr_value):
        return None

    # Score is now a quality ranking on top of hard gates, not a loose gate.
    score = 70
    confirmations = 4
    conflicts = 0
    reasons = list(entry_reasons)
    reasons.append(pullback_reason)
    reasons.append("5M and 15M trend strength agree with direction.")
    reasons.append("MACD confirms direction.")

    if direction == "CALL":
        if 56 <= rsi_value <= 63:
            score += 8
            confirmations += 1
            reasons.append("RSI is in the preferred CALL zone.")
        else:
            score += 4
    else:
        if 36 <= rsi_value <= 44:
            score += 8
            confirmations += 1
            reasons.append("RSI is in the preferred PUT zone.")
        else:
            score += 4

    if two_candle_confirmation(candles_1m, direction):
        score += 8
        confirmations += 1
        reasons.append("Two-candle momentum confirms the move.")

    mom = candle_momentum(candles_1m)
    if (direction == "CALL" and mom >= 0.20) or (direction == "PUT" and mom <= -0.20):
        score += 6
        confirmations += 1
        reasons.append("Recent 1M momentum confirms direction.")

    sr_ok, sr_reason = near_support_resistance(candles_1m, direction, atr_value)
    if sr_ok:
        score += 3
        reasons.append(sr_reason)

    score = max(0, min(100, int(round(score))))
    confidence = confidence_from_score(score, confirmations, conflicts)
    minimum = MIN_SCORE_CALL if direction == "CALL" else MIN_SCORE_PUT
    if score < minimum or confidence < MIN_CONFIDENCE:
        return None

    return {
        "direction": direction,
        "expiry": V2_EXPIRY_MINUTES,
        "score": score,
        "confidence": confidence,
        "rsi": rsi_value,
        "bias_5m": bias_5m,
        "bias_15m": bias_15m,
        "entry_reference": candles_1m[-1].close,
        "reasons": reasons[:8],
    }


def analyze_symbol(
    iq: IQ_Option,
    symbol: str,
    market_type: str,
) -> Optional[Signal]:
    candles_1m = get_candles(iq, symbol, "1M", CANDLE_COUNT)
    candles_5m = get_candles(iq, symbol, "5M", CANDLE_COUNT)
    candles_15m = get_candles(iq, symbol, "15M", CANDLE_COUNT)

    if any(len(c) < MIN_CANDLES for c in (candles_1m, candles_5m, candles_15m)):
        LOGGER.debug("%s: rejected - insufficient candles.", symbol)
        return None

    for label, candles, seconds in ((
        ("1M", candles_1m, TIMEFRAMES["1M"]),
        ("5M", candles_5m, TIMEFRAMES["5M"]),
        ("15M", candles_15m, TIMEFRAMES["15M"]),
    )):
        age = _iq_server_time(iq) - candles[-1].timestamp
        allowed = max(seconds * 2, MAX_SIGNAL_AGE_SECONDS)
        if age > allowed:
            LOGGER.debug("%s: rejected - %s candles stale (age=%ss).", symbol, label, age)
            return None

    setup = analyze_setup(candles_1m, candles_5m, candles_15m)
    if setup is None:
        LOGGER.debug("%s: rejected - V2 quality gates not met.", symbol)
        return None

    return Signal(
        pair=symbol,
        market_type=market_type,
        direction=setup["direction"],
        expiry_minutes=setup["expiry"],
        score=setup["score"],
        confidence=setup["confidence"],
        entry_reference=setup["entry_reference"],
        rsi=setup["rsi"],
        bias_5m=setup["bias_5m"],
        bias_15m=setup["bias_15m"],
        reasons=setup["reasons"],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ============================================================
# SIGNAL SELECTION
# ============================================================

def select_best_signal(
    iq: IQ_Option,
    state: Dict[str, Any],
) -> Optional[Signal]:

    if daily_limit_reached(state):
        LOGGER.info(
            "Daily limit reached: %s/%s.",
            state["signal_count"],
            MAX_DAILY_SIGNALS,
        )
        return None

    candidates: List[Signal] = []

    for regular_pair in REGULAR_PAIRS:

        actual_symbol, market_type = market_status(
            iq,
            regular_pair,
        )

        if actual_symbol is None:
            LOGGER.debug(
                "%s unavailable; skipping.",
                regular_pair,
            )
            continue

        LOGGER.debug(
            "%s -> %s (%s) available.",
            regular_pair,
            actual_symbol,
            market_type,
        )

        try:
            signal = analyze_symbol(
                iq,
                actual_symbol,
                market_type,
            )

            if signal is not None:
                candidates.append(signal)

        except Exception as exc:
            LOGGER.exception(
                "Analysis failed for %s: %s",
                actual_symbol,
                exc,
            )

        time.sleep(PAIR_SCAN_DELAY_SECONDS)

    if not candidates:
        return None

    candidates.sort(
        key=lambda s: (
            s.score,
            s.confidence,
            1 if s.market_type == "REGULAR" else 0,
        ),
        reverse=True,
    )

    best = candidates[0]

    now = time.time()
    last_signal = float(
        state.get("last_signal_epoch", 0.0)
    )

    if now - last_signal < SIGNAL_COOLDOWN_SECONDS:
        LOGGER.info(
            "Signal cooldown active; no new signal sent."
        )
        return None

    LOGGER.info(
        "BEST SIGNAL: %s %s %s | score=%s confidence=%.2f",
        best.pair,
        best.market_type,
        best.direction,
        best.score,
        best.confidence,
    )

    return best


# ============================================================
# VALIDATION / RECORDING
# ============================================================

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
            "Missing required environment variables: "
            + ", ".join(missing)
        )

    if ACCOUNT_MODE != "PRACTICE":
        raise RuntimeError(
            "ACCOUNT_MODE must be PRACTICE."
        )

    if AUTO_TRADE:
        raise RuntimeError(
            "AUTO_TRADE must be False."
        )


def record_sent_signal(
    state: Dict[str, Any],
    signal: Signal,
) -> None:

    new_count = int(state.get("signal_count", 0)) + 1

    if new_count > MAX_DAILY_SIGNALS:
        raise RuntimeError(
            "Safety check blocked a fifth daily signal."
        )

    state["date"] = nigeria_today()
    state["signal_count"] = new_count
    state["last_signal_epoch"] = time.time()

    state.setdefault("signals", []).append(
        {
            **asdict(signal),
            "sent_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    )

    state["signals"] = state["signals"][-20:]

    save_state(state)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    validate_environment()

    state = load_state()

    if daily_limit_reached(state):
        LOGGER.info(
            "Daily limit already reached: %s/%s.",
            state["signal_count"],
            MAX_DAILY_SIGNALS,
        )
        return

    iq: Optional[IQ_Option] = None

    try:
        iq = connect_iq()

        # Connection test.
        telegram_send(
            "✅ TEST: IQ Option scanner connected successfully."
        )

        signal = select_best_signal(
            iq,
            state,
        )

        if signal is None:
            LOGGER.info(
                "No qualifying high-confidence setup found."
            )
            return

        state = load_state()

        if daily_limit_reached(state):
            LOGGER.warning(
                "Daily limit reached during final safety check."
            )
            return

        message = format_signal(
            signal,
            int(state.get("signal_count", 0)) + 1,
        )

        if telegram_send(message):
            record_sent_signal(
                state,
                signal,
            )

            LOGGER.info(
                "Signal sent: %s %s %s | %s/%s.",
                signal.pair,
                signal.market_type,
                signal.direction,
                state["signal_count"],
                MAX_DAILY_SIGNALS,
            )
        else:
            LOGGER.error(
                "Signal was not recorded because Telegram failed."
            )

    except Exception as exc:
        LOGGER.exception(
            "Scanner failed: %s",
            exc,
        )
        raise

    finally:
        if iq is not None:
            try:
                iq.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
