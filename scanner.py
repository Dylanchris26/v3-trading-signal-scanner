import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np

from datetime import datetime, timedelta, timezone
from iqoptionapi.stable_api import IQ_Option


# ============================================================
# V4 — PROFESSIONAL-STYLE LIVE SIGNAL SCANNER
# ============================================================
#
# PURPOSE
# -------
# Practice/demo market scanner for IQ Option.
#
# Main objectives:
#   • High-quality signals only
#   • Maximum 4 signals per day
#   • Rank pairs instead of blindly signalling every pair
#   • Multi-timeframe confirmation
#   • Adaptive expiry
#   • Duplicate protection
#   • Signal cooldown
#   • Daily/monthly progress tracking
#   • Telegram alerts
#   • Automatic trading OFF
#
# IMPORTANT
# ---------
# The $10 -> $100 objective is a TARGET ONLY.
# It is NOT a guaranteed return and is never used to force trades.
#
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

PAIRS = [
    "NZDJPY",
    "GBPJPY",
    "USDJPY",
    "EURJPY",
]

# History
CANDLE_COUNT = 400

# Scanner frequency
SCAN_INTERVAL_SECONDS = 60

# Maximum number of signals allowed in one UTC/local bot day
MAX_SIGNALS_PER_DAY = 4

# Minimum time between signals from the same pair
PAIR_COOLDOWN_MINUTES = 20

# Minimum time between ANY signals
GLOBAL_COOLDOWN_MINUTES = 5

# Signal quality
MIN_SCORE = 9
MIN_SEPARATION = 3

# Minimum confidence
MIN_CONFIDENCE = 72

# Monthly tracking target
STARTING_TARGET = 10.0
MONTHLY_TARGET = 100.0

# Demo/practice only
ACCOUNT_MODE = "PRACTICE"

# Never automatically place trades
AUTO_TRADE = False

# Persistent local state
STATE_FILE = "v4_state.json"


# ============================================================
# LOGGING
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

missing = [
    name
    for name, value in required.items()
    if not value
]

if missing:
    raise RuntimeError(
        "Missing GitHub Secrets: " +
        ", ".join(missing)
    )


# ============================================================
# GLOBAL IQ OPTION OBJECT
# ============================================================

Iq = None


# ============================================================
# TIME HELPERS
# ============================================================

def now_utc():
    return datetime.now(timezone.utc)


def now_string():
    return now_utc().strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def month_key():
    return now_utc().strftime("%Y-%m")


def day_key():
    return now_utc().strftime("%Y-%m-%d")


# ============================================================
# STATE MANAGEMENT
# ============================================================

def default_state():

    return {
        "month": month_key(),
        "day": day_key(),

        "signals_today": 0,

        "starting_balance": STARTING_TARGET,

        "tracked_balance": STARTING_TARGET,

        "wins": 0,
        "losses": 0,

        "signals_total": 0,

        "last_signal_time": None,

        "last_signal_by_pair": {},

        "signal_history": [],
    }


def load_state():

    if not os.path.exists(STATE_FILE):
        return default_state()

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            state = json.load(file)

    except Exception as e:

        print(
            "⚠️ State file could not be read:",
            e
        )

        return default_state()

    defaults = default_state()

    for key, value in defaults.items():

        if key not in state:
            state[key] = value

    # New month
    if state.get("month") != month_key():

        state = default_state()

    # New day
    elif state.get("day") != day_key():

        state["day"] = day_key()
        state["signals_today"] = 0

    return state


def save_state(state):

    try:

        with open(
            STATE_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                state,
                file,
                indent=4,
                default=str
            )

    except Exception as e:

        print(
            "⚠️ Could not save state:",
            e
        )


STATE = load_state()


# ============================================================
# RESET DAILY STATE
# ============================================================

def refresh_day_state():

    global STATE

    current_day = day_key()
    current_month = month_key()

    if STATE.get("month") != current_month:

        STATE = default_state()

        save_state(STATE)

        return

    if STATE.get("day") != current_day:

        STATE["day"] = current_day
        STATE["signals_today"] = 0

        save_state(STATE)


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    try:

        url = (
            f"https://api.telegram.org/"
            f"bot{BOT_TOKEN}/sendMessage"
        )

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

            print(
                "📨 Telegram alert sent"
            )

            return True

        print(
            "❌ Telegram error:",
            response.text
        )

        return False

    except Exception as e:

        print(
            "❌ Telegram exception:",
            e
        )

        return False


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value):

    try:

        result = float(value)

        if np.isfinite(result):
            return result

    except Exception:
        pass

    return None


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema9"] = (
        df["close"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["ema21"] = (
        df["close"]
        .ewm(span=21, adjust=False)
        .mean()
    )

    df["ema20"] = (
        df["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df["ema100"] = (
        df["close"]
        .ewm(span=100, adjust=False)
        .mean()
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        df["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["macd"] = ema12 - ema26

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    df["rsi"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

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

    df["upper_wick"] = (
        df["high"] -
        df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) -
        df["low"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = df["close"].shift(1)

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

    # --------------------------------------------------------
    # ATR average
    # --------------------------------------------------------

    df["atr_mean"] = (
        df["atr"]
        .rolling(30)
        .mean()
    )

    # --------------------------------------------------------
    # Recent support/resistance
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Short-term price momentum
    # --------------------------------------------------------

    df["momentum_3"] = (
        df["close"] -
        df["close"].shift(3)
    )

    df["momentum_5"] = (
        df["close"] -
        df["close"].shift(5)
    )

    # --------------------------------------------------------
    # EMA slope
    # --------------------------------------------------------

    df["ema21_slope"] = (
        df["ema21"] -
        df["ema21"].shift(3)
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

    # IQ Option API normally uses:
    # max = high
    # min = low

    if "max" in df.columns:
        df["high"] = df["max"]

    if "min" in df.columns:
        df["low"] = df["min"]

    required_columns = [
        "open",
        "close",
        "high",
        "low",
    ]

    for col in required_columns:

        if col not in df.columns:

            raise RuntimeError(
                f"{pair}: missing candle field "
                f"'{col}'"
            )

    # Timestamp

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

    # Remove currently-forming 1m candle

    current_minute = (
        pd.Timestamp.now(
            tz="UTC"
        ).floor("min")
    )

    df = df[
        df.index <
        current_minute
    ]

    if len(df) < 150:

        raise RuntimeError(
            f"{pair}: insufficient completed "
            f"1m candles"
        )

    return df


# ============================================================
# BUILD HIGHER TIMEFRAMES
# ============================================================

def make_resampled_data(
    df,
    timeframe
):

    result = df[
        [
            "open",
            "high",
            "low",
            "close",
        ]
    ].resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })

    if timeframe == "5min":

        current_period = (
            pd.Timestamp.now(
                tz="UTC"
            ).floor("5min")
        )

    elif timeframe == "15min":

        current_period = (
            pd.Timestamp.now(
                tz="UTC"
            ).floor("15min")
        )

    else:

        current_period = None

    if current_period is not None:

        result = result[
            result.index <
            current_period
        ]

    result = result.dropna()

    return result


# ============================================================
# MARKET REGIME
# ============================================================

def determine_regime(five):

    latest = five.iloc[-1]

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
        ema100,
    ]:

        return "UNKNOWN"

    if (
        ema20 > ema50
        and
        ema50 > ema100
    ):

        return "STRONG_UPTREND"

    if (
        ema20 < ema50
        and
        ema50 < ema100
    ):

        return "STRONG_DOWNTREND"

    if ema20 > ema50:

        return "UPTREND"

    if ema20 < ema50:

        return "DOWNTREND"

    return "RANGE"


# ============================================================
# ADAPTIVE EXPIRY
# ============================================================

def determine_expiry(
    signal,
    one,
    five,
    fifteen
):

    latest_1 = one.iloc[-1]
    latest_5 = five.iloc[-1]
    latest_15 = fifteen.iloc[-1]

    atr = safe_float(
        latest_1["atr"]
    )

    atr_mean = safe_float(
        latest_1["atr_mean"]
    )

    body_ratio = safe_float(
        latest_1["body_ratio"]
    )

    if any(
        value is None
        for value in [
            atr,
            atr_mean,
            body_ratio,
        ]
    ):

        return 5

    regime_15 = determine_regime(
        fifteen
    )

    regime_5 = determine_regime(
        five
    )

    volatility_ratio = (
        atr / atr_mean
        if atr_mean > 0
        else 1
    )

    # Strong alignment + healthy momentum
    if (
        regime_15 in [
            "STRONG_UPTREND",
            "STRONG_DOWNTREND",
        ]
        and
        regime_5 in [
            "STRONG_UPTREND",
            "STRONG_DOWNTREND",
        ]
        and
        body_ratio >= 0.60
        and
        0.80 <= volatility_ratio <= 1.80
    ):

        return 5

    # Very short-term strong continuation
    if (
        body_ratio >= 0.70
        and
        volatility_ratio >= 1.10
    ):

        return 3

    # Slower / weaker environment
    if (
        body_ratio < 0.55
        or
        volatility_ratio < 0.75
    ):

        return 5

    return 5


# ============================================================
# SIGNAL CONFIDENCE
# ============================================================

def calculate_confidence(
    score,
    opposing_score,
    confirmations
):

    # Base confidence
    confidence = 50

    score_bonus = min(
        max(score - MIN_SCORE, 0) * 3,
        12
    )

    separation_bonus = min(
        max(
            score -
            opposing_score -
            MIN_SEPARATION,
            0
        ) * 3,
        12
    )

    confirmation_bonus = min(
        confirmations * 4,
        20
    )

    confidence += (
        score_bonus +
        separation_bonus +
        confirmation_bonus
    )

    return int(
        min(confidence, 98)
    )


# ============================================================
# SIGNAL ELIGIBILITY
# ============================================================

def pair_is_on_cooldown(pair):

    last_times = STATE.get(
        "last_signal_by_pair",
        {}
    )

    value = last_times.get(pair)

    if not value:
        return False

    try:

        last_time = datetime.fromisoformat(
            value
        )

        elapsed = (
            now_utc() -
            last_time
        ).total_seconds() / 60

        return (
            elapsed <
            PAIR_COOLDOWN_MINUTES
        )

    except Exception:

        return False


def global_is_on_cooldown():

    value = STATE.get(
        "last_signal_time"
    )

    if not value:
        return False

    try:

        last_time = datetime.fromisoformat(
            value
        )

        elapsed = (
            now_utc() -
            last_time
        ).total_seconds() / 60

        return (
            elapsed <
            GLOBAL_COOLDOWN_MINUTES
        )

    except Exception:

        return False


# ============================================================
# DUPLICATE PROTECTION
# ============================================================

def signal_already_sent(
    pair,
    signal,
    timestamp
):

    history = STATE.get(
        "signal_history",
        []
    )

    timestamp_text = str(
        timestamp
    )

    for item in history:

        if (
            item.get("pair") == pair
            and
            item.get("signal") == signal
            and
            item.get("timestamp") ==
            timestamp_text
        ):

            return True

    return False


# ============================================================
# ANALYZE ONE PAIR
# ============================================================

def analyze_pair(pair):

    one_min = get_market_data(
        pair
    )

    five_min = make_resampled_data(
        one_min,
        "5min"
    )

    fifteen_min = make_resampled_data(
        one_min,
        "15min"
    )

    if len(five_min) < 100:

        return {
            "status": "WAIT",
            "reason": "Not enough 5m candles",
        }

    if len(fifteen_min) < 60:

        return {
            "status": "WAIT",
            "reason": "Not enough 15m candles",
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

    latest_1 = one.iloc[-1]
    latest_5 = five.iloc[-1]
    latest_15 = fifteen.iloc[-1]

    # --------------------------------------------------------
    # CORE VALUES
    # ------------------------------------------
