import os
import time
import logging
import requests
import pandas as pd
import numpy as np

from iqoptionapi.stable_api import IQ_Option


# ============================================================
# SETTINGS
# ============================================================

PAIRS = [
    "NZDJPY",
    "GBPJPY",
    "USDJPY",
    "EURJPY",
]

CANDLE_COUNT = 400

EXPIRY_MINUTES = 5

MIN_SCORE = 8
MIN_SEPARATION = 3

ACCOUNT_MODE = "PRACTICE"


# ============================================================
# QUIET IQ OPTION LOGGING
# ============================================================

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("iqoptionapi").setLevel(logging.WARNING)
logging.getLogger("iqoptionapi.ws").setLevel(logging.WARNING)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

IQ_EMAIL = os.environ.get("IQ_EMAIL")
IQ_PASSWORD = os.environ.get("IQ_PASSWORD")

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")


required = {
    "IQ_EMAIL": IQ_EMAIL,
    "IQ_PASSWORD": IQ_PASSWORD,
    "BOT_TOKEN": BOT_TOKEN,
    "CHAT_ID": CHAT_ID,
}

missing = [name for name, value in required.items() if not value]

if missing:
    raise RuntimeError(
        "Missing GitHub Secrets: " + ", ".join(missing)
    )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message,
            },
            timeout=15,
        )

        if response.status_code == 200:
            print("📨 Telegram alert sent")
            return True

        print("❌ Telegram error:", response.text)
        return False

    except Exception as e:
        print("❌ Telegram exception:", e)
        return False


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # EMA
    df["ema9"] = df["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # MACD
    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # Candle structure
    df["range"] = (
        df["high"] -
        df["low"]
    )

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(0, np.nan)
    )

    # ATR
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    # Recent support / resistance
    df["recent_high"] = (
        df["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["recent_low"] = (
        df["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    return df


# ============================================================
# RAW CANDLE DATA
# ============================================================

def get_market_data(pair):

    candles = Iq.get_candles(
        pair,
        60,
        CANDLE_COUNT,
        time.time()
    )

    if not candles:
        raise RuntimeError(
            f"No candle data for {pair}"
        )

    df = pd.DataFrame(candles)

    # IQ Option normally uses:
    # max = high
    # min = low

    if "max" in df.columns:
        df["high"] = df["max"]

    if "min" in df.columns:
        df["low"] = df["min"]

    # Some API versions may already provide
    # high / low.

    required_columns = [
        "open",
        "close",
        "high",
        "low",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise RuntimeError(
                f"{pair}: missing candle field '{col}'"
            )

    # Timestamp
    if "from" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["from"],
            unit="s"
        )

    elif "at" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["at"],
            unit="s"
        )

    else:
        raise RuntimeError(
            f"{pair}: no candle timestamp"
        )

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        "timestamp"
    )

    df = df.set_index(
        "timestamp"
    )

    # Numeric conversion
    for col in [
        "open",
        "close",
        "high",
        "low",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "open",
            "close",
            "high",
            "low",
        ]
    )

    # Remove currently-forming 1-minute candle
    current_minute = pd.Timestamp.now().floor("min")

    df = df[
        df.index < current_minute
    ]

    if len(df) < 100:
        raise RuntimeError(
            f"{pair}: insufficient completed 1m candles"
        )

    return df


# ============================================================
# BUILD 5-MINUTE DATA
# ============================================================

def make_5m_data(df):

    five = df[
        [
            "open",
            "high",
            "low",
            "close",
        ]
    ].resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })

    # Remove incomplete current 5-minute candle
    current_5m = pd.Timestamp.now().floor("5min")

    five = five[
        five.index < current_5m
    ]

    five = five.dropna()

    return five


# ============================================================
# ANALYZE ONE PAIR
# ============================================================

def analyze_pair(pair):

    one_min = get_market_data(pair)

    five_min = make_5m_data(one_min)

    if len(five_min) < 60:
        return {
            "status": "WAIT",
            "reason": "Not enough 5m candles",
        }

    one = add_indicators(
        one_min
    )

    five = add_indicators(
        five_min
    )

    latest_1 = one.iloc[-1]
    latest_5 = five.iloc[-1]

    # --------------------------------------------------------
    # VALUES
    # --------------------------------------------------------

    rsi = float(latest_1["rsi"])

    price = float(latest_1["close"])

    body_ratio = float(
        latest_1["body_ratio"]
    )

    atr = float(
        latest_1["atr"]
    )

    candle_range = float(
        latest_1["range"]
    )

    # Safety against invalid calculations
    if any(
        pd.isna(x)
        for x in [
            rsi,
            price,
            body_ratio,
            atr,
            candle_range,
        ]
    ):
        return {
            "status": "WAIT",
            "reason": "Indicators not ready",
        }

    # --------------------------------------------------------
    # VOLATILITY FILTER
    # --------------------------------------------------------

    if atr <= 0:
        return {
            "status": "WAIT",
            "reason": "Invalid ATR",
        }

    if candle_range < (
        0.50 * atr
    ):
        return {
            "status": "WAIT",
            "reason": "Weak volatility",
            "call": 0,
            "put": 0,
            "rsi": rsi,
        }

    # --------------------------------------------------------
    # CANDLE STRENGTH
    # --------------------------------------------------------

    if body_ratio < 0.45:
        return {
            "status": "WAIT",
            "reason": "Weak candle",
            "call": 0,
            "put": 0,
            "rsi": rsi,
        }

    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    call_score = 0
    put_score = 0

    reasons_call = []
    reasons_put = []

    # ========================================================
    # 5-MINUTE TREND
    # ========================================================

    if (
        latest_5["ema20"] >
        latest_5["ema50"]
    ):
        call_score += 3
        reasons_call.append(
            "5m uptrend"
        )

    elif (
        latest_5["ema20"] <
        latest_5["ema50"]
    ):
        put_score += 3
        reasons_put.append(
            "5m downtrend"
        )

    # ========================================================
    # 1-MINUTE TREND
    # ========================================================

    if (
        latest_1["ema9"] >
        latest_1["ema21"]
    ):
        call_score += 2
        reasons_call.append(
            "1m bullish"
        )

    elif (
        latest_1["ema9"] <
        latest_1["ema21"]
    ):
        put_score += 2
        reasons_put.append(
            "1m bearish"
        )

    # ========================================================
    # MACD
    # ========================================================

    if (
        latest_1["macd"] >
        latest_1["macd_signal"]
        and
        latest_1["macd_hist"] > 0
    ):
        call_score += 2
        reasons_call.append(
            "MACD bullish"
        )

    elif (
        latest_1["macd"] <
        latest_1["macd_signal"]
        and
        latest_1["macd_hist"] < 0
    ):
        put_score += 2
        reasons_put.append(
            "MACD bearish"
        )

    # ========================================================
    # RSI
    # ========================================================

    if 52 <= rsi <= 68:
        call_score += 1
        reasons_call.append(
            "RSI bullish zone"
        )

    elif 32 <= rsi <= 48:
        put_score += 1
        reasons_put.append(
            "RSI bearish zone"
        )

    # ========================================================
    # CANDLE DIRECTION
    # ========================================================

    if latest_1["close"] > latest_1["open"]:
        call_score += 1
        reasons_call.append(
            "bullish candle"
        )

    elif latest_1["close"] < latest_1["open"]:
        put_score += 1
        reasons_put.append(
            "bearish candle"
        )

    # ========================================================
    # SUPPORT / RESISTANCE FILTER
    # ========================================================

    recent_high = latest_1[
        "recent_high"
    ]

    recent_low = latest_1[
        "recent_low"
    ]

    # Don't CALL directly into recent resistance
    if (
        not pd.isna(recent_high)
        and price >= recent_high
    ):
        call_score = min(
            call_score,
            MIN_SCORE - 1
        )

    # Don't PUT directly into recent support
    if (
        not pd.isna(recent_low)
        and price <= recent_low
    ):
        put_score = min(
            put_score,
            MIN_SCORE - 1
        )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    separation = abs(
        call_score -
        put_score
    )

    signal = None
    reasons = []

    # Strong CALL
    if (
        call_score >= MIN_SCORE
        and
        call_score - put_score
        >= MIN_SEPARATION
        and
        latest_5["ema20"] >
        latest_5["ema50"]
        and
        latest_1["ema9"] >
        latest_1["ema21"]
    ):
        signal = "CALL"
        reasons = reasons_call

    # Strong PUT
    elif (
        put_score >= MIN_SCORE
        and
        put_score - call_score
        >= MIN_SEPARATION
        and
        latest_5["ema20"] <
        latest_5["ema50"]
        and
        latest_1["ema9"] <
        latest_1["ema21"]
    ):
        signal = "PUT"
        reasons = reasons_put

    return {
        "status": "SIGNAL" if signal else "WAIT",
        "signal": signal,
        "call": call_score,
        "put": put_score,
        "rsi": rsi,
        "price": price,
        "body_ratio": body_ratio,
        "timestamp": one.index[-1],
        "reasons": reasons,
        "separation": separation,
    }


# ============================================================
# CONNECT TO IQ OPTION
# ============================================================

print("=" * 72)
print("🚀 VERSION 3 GITHUB LIVE SIGNAL SCANNER")
print("=" * 72)

print("🧪 Account: PRACTICE / DEMO")
print("📡 Live market data: ON")
print(
    "📊 Markets:",
    ", ".join(PAIRS)
)
print(
    f"📚 History: {CANDLE_COUNT} × 1-minute candles"
)
print(
    f"⏱️ Expiry: {EXPIRY_MINUTES} minutes"
)
print("📨 Telegram: ON")
print("💰 Automatic trading: OFF")
print("=" * 72)


# ============================================================
# IQ OPTION CONNECTION
# ============================================================

print()
print("🔐 Connecting to IQ Option...")

Iq = IQ_Option(
    IQ_EMAIL,
    IQ_PASSWORD
)

connected, reason = Iq.connect()

if not connected:
    raise RuntimeError(
        f"IQ Option connection failed: {reason}"
    )

print("✅ IQ Option connection successful")

Iq.change_balance(
    ACCOUNT_MODE
)

print(
    "🧪 Account mode confirmed:",
    ACCOUNT_MODE
)


# ============================================================
# TELEGRAM TEST
# ============================================================

print()
print("📨 Testing Telegram connection...")

telegram_ok = send_telegram(
    "🟢 V3 SCANNER ONLINE\n\n"
    "📡 GitHub Actions scanner connected.\n"
    "🧪 Mode: PRACTICE / DEMO\n"
    "💰 Automatic trading: OFF"
)

if not telegram_ok:
    raise RuntimeError(
        "Telegram connection failed"
    )


# ============================================================
# SCAN MARKETS
# ============================================================

print()
print("=" * 72)
print(
    "🔎 MARKET SCAN",
    time.strftime("%Y-%m-%d %H:%M:%S")
)
print("=" * 72)

signals_found = []

for pair in PAIRS:

    try:

        result = analyze_pair(pair)

        if result["status"] == "SIGNAL":

            signal = result["signal"]

            print(
                f"🚨 {pair} | "
                f"{signal} | "
                f"CALL {result['call']} | "
                f"PUT {result['put']} | "
                f"RSI {result['rsi']:.1f}"
            )

            signals_found.append(
                (pair, result)
            )

        else:

            print(
                f"⏸️ {pair} | "
                f"CALL {result.get('call', 0)} | "
                f"PUT {result.get('put', 0)} | "
                f"RSI {result.get('rsi', 0):.1f} | "
                f"{result.get('reason', 'No high-quality signal')}"
            )

    except Exception as e:

        print(
            f"❌ {pair} | ERROR | {e}"
        )


# ============================================================
# SEND SIGNALS
# ============================================================

print()
print("=" * 72)

if not signals_found:

    print(
        "💤 No high-quality signal this scan."
    )

else:

    print(
        f"🚨 {len(signals_found)} "
        "high-quality signal(s) found."
    )

    for pair, result in signals_found:

        direction_emoji = (
            "🟢" if result["signal"] == "CALL"
            else "🔴"
        )

        reasons_text = "\n".join(
            f"• {reason}"
            for reason in result["reasons"]
        )

        message = (
            f"🚨 V3 TRADING SIGNAL\n\n"
            f"{direction_emoji} "
            f"{result['signal']}\n"
            f"📊 Pair: {pair}\n"
            f"💰 Price: {result['price']}\n"
            f"⏱️ Expiry: "
            f"{EXPIRY_MINUTES} minutes\n\n"
            f"📈 CALL score: "
            f"{result['call']}\n"
            f"📉 PUT score: "
            f"{result['put']}\n"
            f"📊 RSI: "
            f"{result['rsi']:.1f}\n\n"
            f"🧠 Confirmation:\n"
            f"{reasons_text}\n\n"
            f"🧪 PRACTICE / DEMO ONLY\n"
            f"💰 Automatic trading: OFF"
        )

        send_telegram(
            message
        )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 72)
print("✅ SCAN COMPLETE")
print("💰 No automatic trades were placed.")
print("=" * 72)
