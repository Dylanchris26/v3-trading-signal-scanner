import os
import time
import json
import traceback
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
from iqoptionapi.stable_api import IQ_Option


# ============================================================
# V4.1 CONFIG
# ============================================================

PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "EURJPY",
    "GBPJPY",
    "NZDJPY",
]

TIMEFRAMES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
}

CANDLE_COUNT = 250

SCAN_INTERVAL = 30
MAX_SIGNALS_PER_DAY = 4

MIN_CONFIDENCE = 75
MIN_SCORE = 10
MIN_SEPARATION = 3

ACCOUNT_MODE = "PRACTICE"
AUTO_TRADE = False

STATE_FILE = "v41_state.json"


# ============================================================
# SECRETS
# ============================================================

IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")


# ============================================================
# TELEGRAM
# ============================================================

def telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials missing", flush=True)
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message,
            },
            timeout=20,
        )

        print(
            f"Telegram HTTP {response.status_code}",
            flush=True
        )

        if response.status_code != 200:
            print(response.text, flush=True)

        return response.status_code == 200

    except Exception as e:
        print(f"Telegram error: {e}", flush=True)
        return False


# ============================================================
# STATE
# ============================================================

def current_day():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_state():
    default = {
        "date": current_day(),
        "signals_today": 0,
        "last_signals": {},
        "history": [],
    }

    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                state = json.load(f)

            if state.get("date") == current_day():
                return state

    except Exception as e:
        print(f"State warning: {e}", flush=True)

    return default


def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"State save warning: {e}", flush=True)


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    return result.fillna(50)


def atr(df, period=14):
    high_low = df["high"] - df["low"]

    high_close = (
        df["high"] - df["close"].shift()
    ).abs()

    low_close = (
        df["low"] - df["close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close,
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(period).mean()


def add_indicators(df):
    df = df.copy()

    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)

    df["rsi"] = rsi(df["close"])

    fast = ema(df["close"], 12)
    slow = ema(df["close"], 26)

    df["macd"] = fast - slow
    df["macd_signal"] = ema(df["macd"], 9)

    df["atr"] = atr(df)

    return df


# ============================================================
# CANDLES
# ============================================================

def get_candles(iq, pair, timeframe, count):
    try:
        candles = iq.get_candles(
            pair,
            timeframe,
            count,
            time.time()
        )

        if not candles:
            print(
                f"{pair} {timeframe}: no candles",
                flush=True
            )
            return None

        df = pd.DataFrame(candles)

        if "min" not in df.columns or "max" not in df.columns:
            return None

        df = df.rename(
            columns={
                "min": "low",
                "max": "high",
            }
        )

        needed = [
            "open",
            "high",
            "low",
            "close",
        ]

        df = df[needed].copy()

        for column in needed:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna()

        if len(df) < 120:
            return None

        return df.reset_index(drop=True)

    except Exception as e:
        print(
            f"{pair} {timeframe} candle error: {e}",
            flush=True
        )
        return None


# ============================================================
# TIMEFRAME TREND
# ============================================================

def timeframe_bias(df):
    if df is None or len(df) < 100:
        return "NEUTRAL", 0

    df = add_indicators(df)
    c = df.iloc[-1]

    bullish = 0
    bearish = 0

    if c["ema9"] > c["ema21"]:
        bullish += 1
    else:
        bearish += 1

    if c["ema21"] > c["ema50"]:
        bullish += 1
    else:
        bearish += 1

    if c["ema50"] > c["ema100"]:
        bullish += 1
    else:
        bearish += 1

    if c["macd"] > c["macd_signal"]:
        bullish += 1
    else:
        bearish += 1

    if bullish >= 3:
        return "CALL", bullish

    if bearish >= 3:
        return "PUT", bearish

    return "NEUTRAL", max(bullish, bearish)


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def market_levels(df, lookback=50):
    recent = df.tail(lookback)

    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    return support, resistance


# ============================================================
# 1-MINUTE SETUP
# ============================================================

def analyze_entry(df):
    if df is None or len(df) < 120:
        return None

    df = add_indicators(df)

    c = df.iloc[-1]
    p = df.iloc[-2]

    call_score = 0
    put_score = 0

    call_reasons = []
    put_reasons = []

    # --------------------------------------------------------
    # EMA TREND
    # --------------------------------------------------------

    if c["ema9"] > c["ema21"]:
        call_score += 2
        call_reasons.append("1m EMA bullish")

    if c["ema9"] < c["ema21"]:
        put_score += 2
        put_reasons.append("1m EMA bearish")

    # --------------------------------------------------------
    # MEDIUM TREND
    # --------------------------------------------------------

    if c["ema21"] > c["ema50"]:
        call_score += 2
        call_reasons.append("1m medium trend bullish")

    if c["ema21"] < c["ema50"]:
        put_score += 2
        put_reasons.append("1m medium trend bearish")

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if c["macd"] > c["macd_signal"]:
        call_score += 2
        call_reasons.append("MACD bullish")

    if c["macd"] < c["macd_signal"]:
        put_score += 2
        put_reasons.append("MACD bearish")

    # Fresh cross gets an additional point.
    if (
        c["macd"] > c["macd_signal"]
        and p["macd"] <= p["macd_signal"]
    ):
        call_score += 1
        call_reasons.append("Fresh MACD bullish cross")

    if (
        c["macd"] < c["macd_signal"]
        and p["macd"] >= p["macd_signal"]
    ):
        put_score += 1
        put_reasons.append("Fresh MACD bearish cross")

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    r = float(c["rsi"])

    if 52 <= r <= 67:
        call_score += 2
        call_reasons.append(
            f"RSI supportive ({r:.1f})"
        )

    elif 67 < r <= 72:
        call_score += 1

    if 33 <= r <= 48:
        put_score += 2
        put_reasons.append(
            f"RSI supportive ({r:.1f})"
        )

    elif 28 <= r < 33:
        put_score += 1

    # Do not chase extremes.
    if r >= 75:
        call_score -= 3

    if r <= 25:
        put_score -= 3

    # --------------------------------------------------------
    # CANDLE QUALITY
    # --------------------------------------------------------

    candle_range = float(
        c["high"] - c["low"]
    )

    if candle_range > 0:

        body = abs(
            float(c["close"] - c["open"])
        )

        body_ratio = body / candle_range

        if body_ratio >= 0.55:

            if c["close"] > c["open"]:
                call_score += 2
                call_reasons.append(
                    "Strong bullish candle"
                )

            elif c["close"] < c["open"]:
                put_score += 2
                put_reasons.append(
                    "Strong bearish candle"
                )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    last_three = df.tail(3)

    rising_closes = (
        last_three["close"].iloc[-1]
        > last_three["close"].iloc[0]
    )

    if rising_closes:
        call_score += 1
        call_reasons.append("Short-term upward momentum")

    else:
        put_score += 1
        put_reasons.append("Short-term downward momentum")

    # --------------------------------------------------------
    # ATR / VOLATILITY
    # --------------------------------------------------------

    current_atr = float(c["atr"])

    if not np.isfinite(current_atr) or current_atr <= 0:
        return None

    recent_ranges = (
        df["high"] - df["low"]
    ).tail(20)

    median_range = float(
        recent_ranges.median()
    )

    if median_range <= 0:
        return None

    volatility_ratio = current_atr / median_range

    # Extremely dead markets are rejected.
    if volatility_ratio < 0.65:
        return None

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    support, resistance = market_levels(df)

    price = float(c["close"])

    distance_to_resistance = (
        resistance - price
    )

    distance_to_support = (
        price - support
    )

    # Don't blindly buy directly underneath resistance.
    if distance_to_resistance > 0:

        if distance_to_resistance < current_atr * 0.35:
            call_score -= 2

    # Don't blindly sell directly above support.
    if distance_to_support > 0:

        if distance_to_support < current_atr * 0.35:
            put_score -= 2

    # --------------------------------------------------------
    # CHOOSE DIRECTION
    # --------------------------------------------------------

    if call_score > put_score:
        direction = "CALL"
        score = call_score
        opposing = put_score
        reasons = call_reasons

    elif put_score > call_score:
        direction = "PUT"
        score = put_score
        opposing = call_score
        reasons = put_reasons

    else:
        return None

    separation = score - opposing

    if score < MIN_SCORE:
        return None

    if separation < MIN_SEPARATION:
        return None

    return {
        "direction": direction,
        "score": score,
        "opposing_score": opposing,
        "separation": separation,
        "rsi": r,
        "atr": current_atr,
        "volatility_ratio": volatility_ratio,
        "price": price,
        "support": support,
        "resistance": resistance,
        "reasons": reasons,
    }


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    entry,
    bias_5m,
    strength_5m,
    bias_15m,
    strength_15m,
):
    direction = entry["direction"]

    confidence = 50.0

    # Entry quality.
    confidence += min(
        18,
        entry["score"] * 1.5
    )

    # Score separation.
    confidence += min(
        8,
        entry["separation"] * 2
    )

    # 5m confirmation.
    if bias_5m == direction:
        confidence += 10

    elif bias_5m != "NEUTRAL":
        confidence -= 8

    # 15m confirmation.
    if bias_15m == direction:
        confidence += 10

    elif bias_15m != "NEUTRAL":
        confidence -= 10

    # Strength of higher timeframe trends.
    if bias_5m == direction and strength_5m >= 4:
        confidence += 3

    if bias_15m == direction and strength_15m >= 4:
        confidence += 3

    # Keep confidence honest.
    confidence = max(
        0,
        min(95, confidence)
    )

    return round(confidence)


# ============================================================
# ADAPTIVE EXPIRY
# ============================================================

def choose_expiry(
    confidence,
    entry,
    bias_5m,
    bias_15m,
):
    direction = entry["direction"]

    aligned_5m = bias_5m == direction
    aligned_15m = bias_15m == direction

    # Strong multi-timeframe setup.
    if (
        confidence >= 88
        and aligned_5m
        and aligned_15m
        and entry["score"] >= 14
    ):
        return 3

    # Normal strong setup.
    if confidence >= 80 and aligned_5m:
        return 5

    # Slower trend-following setup.
    if (
        confidence >= 75
        and aligned_5m
        and aligned_15m
    ):
        return 10

    return 5


# ============================================================
# COMPLETE PAIR ANALYSIS
# ============================================================

def analyze_pair(iq, pair):
    print(
        f"Analyzing {pair}...",
        flush=True
    )

    df_1m = get_candles(
        iq,
        pair,
        TIMEFRAMES["1m"],
        CANDLE_COUNT
    )

    df_5m = get_candles(
        iq,
        pair,
        TIMEFRAMES["5m"],
        CANDLE_COUNT
    )

    df_15m = get_candles(
        iq,
        pair,
        TIMEFRAMES["15m"],
        CANDLE_COUNT
    )

    if (
        df_1m is None
        or df_5m is None
        or df_15m is None
    ):
        print(
            f"{pair}: incomplete timeframe data",
            flush=True
        )
        return None

    entry = analyze_entry(df_1m)

    if not entry:
        return None

    bias_5m, strength_5m = timeframe_bias(
        df_5m
    )

    bias_15m, strength_15m = timeframe_bias(
        df_15m
    )

    confidence = calculate_confidence(
        entry,
        bias_5m,
        strength_5m,
        bias_15m,
        strength_15m,
    )

    if confidence < MIN_CONFIDENCE:
        print(
            f"{pair}: rejected — "
            f"{entry['direction']} "
            f"{confidence}% confidence",
            flush=True
        )
        return None

    expiry = choose_expiry(
        confidence,
        entry,
        bias_5m,
        bias_15m,
    )

    # Add higher-timeframe confirmation
    # to the reasons shown to the user.
    reasons = list(entry["reasons"])

    if bias_5m == entry["direction"]:
        reasons.append(
            f"5m trend confirms {entry['direction']}"
        )

    if bias_15m == entry["direction"]:
        reasons.append(
            f"15m trend confirms {entry['direction']}"
        )

    if bias_5m != "NEUTRAL":
        reasons.append(
            f"5m bias: {bias_5m}"
        )

    if bias_15m != "NEUTRAL":
        reasons.append(
            f"15m bias: {bias_15m}"
        )

    return {
        "pair": pair,
        "direction": entry["direction"],
        "score": entry["score"],
        "separation": entry["separation"],
        "confidence": confidence,
        "expiry": expiry,
        "price": entry["price"],
        "rsi": round(entry["rsi"], 1),
        "atr": entry["atr"],
        "bias_5m": bias_5m,
        "bias_15m": bias_15m,
        "reasons": reasons,
    }


# ============================================================
# SIGNAL MESSAGE
# ============================================================

def format_signal(signal):
    icon = (
        "🟢"
        if signal["direction"] == "CALL"
        else "🔴"
    )

    reasons = "\n".join(
        f"• {reason}"
        for reason in signal["reasons"][:7]
    )

    return (
        "🎯 V4.1 PREMIUM SIGNAL\n\n"
        f"💱 Pair: {signal['pair']}\n"
        f"{icon} Direction: {signal['direction']}\n"
        f"⏱ Expiry: {signal['expiry']} minutes\n\n"
        f"📊 Score: {signal['score']}\n"
        f"📐 Separation: {signal['separation']}\n"
        f"🧠 Confidence: {signal['confidence']}%\n"
        f"💰 Entry reference: {signal['price']}\n"
        f"📈 RSI: {signal['rsi']}\n"
        f"5️⃣ 5m trend: {signal['bias_5m']}\n"
        f"1️⃣5️⃣ 15m trend: {signal['bias_15m']}\n\n"
        f"Why:\n{reasons}\n\n"
        "⚠️ Practice/demo signal\n"
        "🤖 Auto-trading: OFF"
    )


# ============================================================
# IQ OPTION CONNECTION
# ============================================================

def connect_iq():
    print(
        "Connecting to IQ Option...",
        flush=True
    )

    iq = IQ_Option(
        IQ_EMAIL,
        IQ_PASSWORD
    )

    success, reason = iq.connect()

    if not success:
        print(
            f"IQ Option connection failed: {reason}",
            flush=True
        )

        telegram(
            "🔴 V4.1 ERROR\n\n"
            "IQ Option connection failed.\n"
            f"Reason: {reason}"
        )

        return None

    print(
        "✅ IQ Option connected",
        flush=True
    )

    try:
        iq.change_balance("PRACTICE")
    except Exception:
        pass

    return iq


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60, flush=True)
    print(
        "🚀 V4.1 PROFESSIONAL SIGNAL ENGINE",
        flush=True
    )
    print("=" * 60, flush=True)

    telegram(
        "🟢 V4.1 SCANNER STARTING\n\n"
        "Telegram: CONNECTED\n"
        "Market engine: STARTING\n"
        "Account: PRACTICE / DEMO\n"
        "Auto-trading: OFF\n\n"
        "Maximum signals today: 4\n"
        "Minimum confidence: 75%"
    )

    if not IQ_EMAIL or not IQ_PASSWORD:
        telegram(
            "🔴 V4.1 ERROR\n\n"
            "IQ Option credentials are missing."
        )
        return

    iq = connect_iq()

    if iq is None:
        return

    state = load_state()

    telegram(
        "🟢 V4.1 SCANNER ONLINE\n\n"
        "IQ Option: CONNECTED\n"
        "1m: Entry timing\n"
        "5m: Primary trend\n"
        "15m: Confirmation\n"
        "Daily limit: 4\n"
        "Auto-trading: OFF\n\n"
        "Waiting for high-quality setups."
    )

    while True:

        try:

            if state["date"] != current_day():
                state = load_state()

            if (
                state["signals_today"]
                >= MAX_SIGNALS_PER_DAY
            ):
                print(
                    "Daily limit reached.",
                    flush=True
      
