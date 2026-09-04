import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np

from datetime import datetime, timezone
from iqoptionapi.stable_api import IQ_Option


# ============================================================
# V4 PROFESSIONAL-STYLE SIGNAL SCANNER
# ============================================================

PAIRS = [
    "NZDJPY",
    "GBPJPY",
    "USDJPY",
    "EURJPY",
]

CANDLE_COUNT = 400

# Scan once every minute
SCAN_INTERVAL_SECONDS = 60

# Maximum number of signals during one scanner session/day
MAX_SIGNALS_PER_DAY = 4

# Prevent repeated signals
PAIR_COOLDOWN_MINUTES = 20
GLOBAL_COOLDOWN_MINUTES = 5

# Signal quality requirements
MIN_SCORE = 9
MIN_SEPARATION = 3
MIN_CONFIDENCE = 72

# Tracking target only — NOT a promise
STARTING_TARGET = 10.0
MONTHLY_TARGET = 100.0

# Safety
ACCOUNT_MODE = "PRACTICE"
AUTO_TRADE = False

STATE_FILE = "v4_state.json"

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("iqoptionapi").setLevel(logging.WARNING)
logging.getLogger("iqoptionapi.ws").setLevel(logging.WARNING)


# ============================================================
# SECRETS
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
# GLOBALS
# ============================================================

Iq = None


# ============================================================
# TIME
# ============================================================

def now_utc():
    return datetime.now(timezone.utc)


def now_string():
    return now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")


def day_key():
    return now_utc().strftime("%Y-%m-%d")


def month_key():
    return now_utc().strftime("%Y-%m")


# ============================================================
# STATE
# ============================================================

def default_state():
    return {
        "day": day_key(),
        "month": month_key(),
        "signals_today": 0,
        "signals_total": 0,
        "wins": 0,
        "losses": 0,
        "starting_balance": STARTING_TARGET,
        "tracked_balance": STARTING_TARGET,
        "last_signal_time": None,
        "last_signal_by_pair": {},
        "signal_history": [],
    }


def load_state():

    if not os.path.exists(STATE_FILE):
        return default_state()

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return default_state()

    defaults = default_state()

    for key, value in defaults.items():
        if key not in state:
            state[key] = value

    if state.get("day") != day_key():
        state["day"] = day_key()
        state["signals_today"] = 0

    if state.get("month") != month_key():
        state = default_state()

    return state


def save_state():

    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(STATE, f, indent=2, default=str)
    except Exception as e:
        print("⚠️ State save error:", e, flush=True)


STATE = load_state()


def refresh_state():

    global STATE

    if STATE.get("month") != month_key():
        STATE = default_state()
        save_state()
        return

    if STATE.get("day") != day_key():
        STATE["day"] = day_key()
        STATE["signals_today"] = 0
        save_state()


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
                "disable_web_page_preview": True,
            },
            timeout=15,
        )

        if response.status_code == 200:
            print("📨 Telegram message sent", flush=True)
            return True

        print(
            f"❌ Telegram failed: HTTP {response.status_code}",
            flush=True
        )

        return False

    except Exception as e:

        print(
            f"❌ Telegram exception: {e}",
            flush=True
        )

        return False


def test_telegram():

    print("📨 Testing Telegram...", flush=True)

    message = (
        "🟢 V4 SCANNER STARTING\n\n"
        "Telegram connection: OK\n"
        "Market scanner: INITIALIZING\n"
        "Account mode: PRACTICE / DEMO\n"
        "Automatic trading: OFF\n\n"
        "The scanner is now preparing the market connection."
    )

    if not send_telegram(message):
        raise RuntimeError(
            "Telegram test failed. Check BOT_TOKEN and CHAT_ID."
        )


# ============================================================
# IQ OPTION
# ============================================================

def connect_iq():

    global Iq

    print("🔐 Connecting to IQ Option...", flush=True)

    Iq = IQ_Option(
        IQ_EMAIL,
        IQ_PASSWORD
    )

    connected, reason = Iq.connect()

    if not connected:

        raise RuntimeError(
            "IQ Option connection failed: " + str(reason)
        )

    print(
        "✅ IQ Option connection successful",
        flush=True
    )

    try:
        Iq.change_balance(ACCOUNT_MODE)
        print(
            f"🧪 Account mode: {ACCOUNT_MODE}",
            flush=True
        )
    except Exception as e:
        print(
            f"⚠️ Could not change balance mode: {e}",
            flush=True
        )


def ensure_connection():

    global Iq

    try:

        if Iq is None:
            connect_iq()
            return

        Iq.get_profile()

    except Exception as e:

        print(
            f"⚠️ Connection check failed: {e}",
            flush=True
        )

        print(
            "🔄 Reconnecting...",
            flush=True
        )

        connect_iq()


# ============================================================
# DATA
# ============================================================

def safe_float(value):

    try:

        result = float(value)

        if np.isfinite(result):
            return result

    except Exception:
        pass

    return None


def get_market_data(pair):

    candles = Iq.get_candles(
        pair,
        60,
        CANDLE_COUNT,
        time.time()
    )

    if not candles:
        raise RuntimeError(
            f"{pair}: no candle data"
        )

    df = pd.DataFrame(candles)

    if "max" in df.columns:
        df["high"] = df["max"]

    if "min" in df.columns:
        df["low"] = df["min"]

    required_columns = [
        "open",
        "close",
        "high",
        "low"
    ]

    for col in required_columns:

        if col not in df.columns:
            raise RuntimeError(
                f"{pair}: missing {col}"
            )

    if "from" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["from"],
            unit="s",
            utc=True
        )

    elif "at" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["at"],
            unit="s",
            utc=True
        )

    else:

        raise RuntimeError(
            f"{pair}: missing timestamp"
        )

    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    )

    # Only completed candles
    current_minute = pd.Timestamp.now(
        tz="UTC"
    ).floor("min")

    df = df[
        df.index < current_minute
    ]

    if len(df) < 150:

        raise RuntimeError(
            f"{pair}: insufficient completed candles"
        )

    return df


def resample_data(df, timeframe):

    result = (
        df[
            [
                "open",
                "high",
                "low",
                "close"
            ]
        ]
        .resample(timeframe)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        })
        .dropna()
    )

    if timeframe == "5min":

        current_period = pd.Timestamp.now(
            tz="UTC"
        ).floor("5min")

        result = result[
            result.index < current_period
        ]

    elif timeframe == "15min":

        current_period = pd.Timestamp.now(
            tz="UTC"
        ).floor("15min")

        result = result[
            result.index < current_period
        ]

    return result


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    close = df["close"]

    df["ema9"] = close.ewm(
        span=9,
        adjust=False
    ).mean()

    df["ema21"] = close.ewm(
        span=21,
        adjust=False
    ).mean()

    df["ema20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["ema100"] = close.ewm(
        span=100,
        adjust=False
    ).mean()

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df[
        "macd"
    ].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["rsi"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

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
        df["range"].replace(
            0,
            np.nan
        )
    )

    previous_close = close.shift(1)

    tr1 = (
        df["high"] -
        df["low"]
    )

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

    df["atr"] = (
        true_range
        .rolling(14)
        .mean()
    )

    df["atr_mean"] = (
        df["atr"]
        .rolling(30)
        .mean()
    )

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

    df["momentum_3"] = (
        close -
        close.shift(3)
    )

    df["momentum_5"] = (
        close -
        close.shift(5)
    )

    df["ema21_slope"] = (
        df["ema21"] -
        df["ema21"].shift(3)
    )

    return df


# ============================================================
# MARKET REGIME
# ============================================================

def determine_regime(df):

    latest = df.iloc[-1]

    ema20 = safe_float(
        latest["ema20"]
    )

    ema50 = safe_float(
        latest["ema50"]
    )

    ema100 = safe_float(
        latest["ema100"]
    )

    if None in [
        ema20,
        ema50,
        ema100
    ]:
        return "UNKNOWN"

    if (
        ema20 > ema50
        and ema50 > ema100
    ):
        return "STRONG_UPTREND"

    if (
        ema20 < ema50
        and ema50 < ema100
    ):
        return "STRONG_DOWNTREND"

    if ema20 > ema50:
        return "UPTREND"

    if ema20 < ema50:
        return "DOWNTREND"

    return "RANGE"


# ============================================================
# EXPIRY
# ============================================================

def determine_expiry(
    one,
    five,
    fifteen
):

    latest = one.iloc[-1]

    atr = safe_float(
        latest["atr"]
    )

    atr_mean = safe_float(
        latest["atr_mean"]
    )

    body_ratio = safe_float(
        latest["body_ratio"]
    )

    if None in [
        atr,
        atr_mean,
        body_ratio
    ]:
        return 5

    volatility = (
        atr / atr_mean
        if atr_mean > 0
        else 1
    )

    regime5 = determine_regime(
        five
    )

    regime15 = determine_regime(
        fifteen
    )

    strong_trend = (
        regime5 in [
            "STRONG_UPTREND",
            "STRONG_DOWNTREND",
        ]
        and
        regime15 in [
            "STRONG_UPTREND",
            "STRONG_DOWNTREND",
        ]
    )

    if (
        strong_trend
        and body_ratio >= 0.60
        and 0.80 <= volatility <= 1.80
    ):
        return 5

    if (
        body_ratio >= 0.70
        and volatility >= 1.10
    ):
        return 3

    return 5


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    score,
    opposing,
    confirmations
):

    confidence = 50

    confidence += min(
        max(score - MIN_SCORE, 0) * 3,
        12
    )

    confidence += min(
        max(
            score -
            opposing -
            MIN_SEPARATION,
            0
        ) * 3,
        12
    )

    confidence += min(
        confirmations * 4,
        20
    )

    return int(
        min(
            confidence,
            98
        )
    )


# ============================================================
# ANALYSIS
# ============================================================

def analyze_pair(pair):

    one_min = get_market_data(
        pair
    )

    five_min = resample_data(
        one_min,
        "5min"
    )

    fifteen_min = resample_data(
        one_min,
        "15min"
    )

    if len(five_min) < 100:

        return {
            "status": "WAIT",
            "reason": "Not enough 5m data",
        }

    if len(fifteen_min) < 60:

        return {
            "status": "WAIT",
            "reason": "Not enough 15m data",
        }

    one = add_indicators(
        one_min
    )

    five = add_indicators(
        five_min
    )

    fifteen = add_indicators(
        fifteen_min
    )

    latest1 = one.iloc[-1]
    latest5 = five.iloc[-1]
    latest15 = fifteen.iloc[-1]

    values = [
        latest1["rsi"],
        latest1["close"],
        latest1["body_ratio"],
        latest1["atr"],
        latest1["atr_mean"],
        latest1["range"],
        latest1["macd_hist"],
        latest1["momentum_3"],
        latest1["momentum_5"],
        latest1["ema21_slope"],
    ]

    if any(
        safe_float(x) is None
        for x in values
    ):

        return {
            "status": "WAIT",
            "reason": "Indicators not ready",
        }

    rsi = float(
        latest1["rsi"]
    )

    price = float(
        latest1["close"]
    )

    body_ratio = float(
        latest1["body_ratio"]
    )

    atr = float(
        latest1["atr"]
    )

    atr_mean = float(
        latest1["atr_mean"]
    )

    candle_range = float(
        latest1["range"]
    )

    macd_hist = float(
        latest1["macd_hist"]
    )

    momentum3 = float(
        latest1["momentum_3"]
    )

    momentum5 = float(
        latest1["momentum_5"]
    )

    slope = float(
        latest1["ema21_slope"]
    )

    if atr <= 0:

        return {
            "status": "WAIT",
            "reason": "Invalid ATR",
        }

    volatility = (
        atr / atr_mean
        if atr_mean > 0
        else 1
    )

    if volatility < 0.60:

        return {
            "status": "WAIT",
            "reason": "Market too quiet",
            "call": 0,
            "put": 0,
            "rsi": rsi,
        }

    if candle_range < 0.50 * atr:

        return {
            "status": "WAIT",
            "reason": "Weak volatility",
            "call": 0,
            "put": 0,
            "rsi": rsi,
        }

    if candle_range > 2.50 * atr:

        return {
            "status": "WAIT",
            "reason": "Abnormal candle",
            "call": 0,
            "put": 0,
            "rsi": rsi,
        }

    if body_ratio < 0.45:

        return {
            "status": "WAIT",
            "reason": "Weak candle body",
            "call": 0,
            "put": 0,
            "rsi": rsi,
        }

    call = 0
    put = 0

    call_reasons = []
    put_reasons = []

    call_confirmations = 0
    put_confirmations = 0

    # 15m trend
    if (
        latest15["ema20"] >
        latest15["ema50"] >
        latest15["ema100"]
    ):

        call += 3
        call_confirmations += 1
        call_reasons.append(
            "15m strong uptrend"
        )

    elif (
        latest15["ema20"] <
        latest15["ema50"] <
        latest15["ema100"]
    ):

        put += 3
        put_confirmations += 1
        put_reasons.append(
            "15m strong downtrend"
        )

    elif (
        latest15["ema20"] >
        latest15["ema50"]
    ):

        call += 2
        call_reasons.append(
            "15m bullish trend"
        )

    elif (
        latest15["ema20"] <
        latest15["ema50"]
    ):

        put += 2
        put_reasons.append(
            "15m bearish trend"
        )

    # 5m trend
    if (
        latest5["ema20"] >
        latest5["ema50"]
    ):

        call += 3
        call_confirmations += 1
        call_reasons.append(
            "5m bullish structure"
        )

    elif (
        latest5["ema20"] <
        latest5["ema50"]
    ):

        put += 3
        put_confirmations += 1
        put_reasons.append(
            "5m bearish structure"
        )

    # 1m structure
    if (
        latest1["ema9"] >
        latest1["ema21"]
    ):

        call += 2
        call_confirmations += 1
        call_reasons.append(
            "1m bullish structure"
        )

    elif (
        latest1["ema9"] <
        latest1["ema21"]
    ):

        put += 2
        put_confirmations += 1
        put_reasons.append(
            "1m bearish structure"
        )

    # MACD
    if (
        latest1["macd"] >
        latest1["macd_signal"]
        and macd_hist > 0
    ):

        call += 2
        call_confirmations += 1
        call_reasons.append(
            "MACD bullish"
        )

    elif (
        latest1["macd"] <
        latest1["macd_signal"]
        and macd_hist < 0
    ):

        put += 2
        put_confirmations += 1
        put_reasons.append(
            "MACD bearish"
        )

    # RSI
    if 52 <= rsi <= 67:

        call += 1
        call_reasons.append(
            "RSI bullish zone"
        )

    elif 33 <= rsi <= 48:

        put += 1
        put_reasons.append(
            "RSI bearish zone"
        )

    # Momentum
    if (
        momentum3 > 0
        and momentum5 > 0
    ):

        call += 2
        call_confirmations += 1
        call_reasons.append(
            "positive momentum"
        )

    elif (
        momentum3 < 0
        and momentum5 < 0
    ):

        put += 2
        put_confirmations += 1
        put_reasons.append(
            "negative momentum"
        )

    # EMA slop
