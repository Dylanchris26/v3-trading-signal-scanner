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
# V4.2 PROFESSIONAL SIGNAL ENGINE
# ============================================================

PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "EURJPY",
    "GBPJPY",
    "NZDJPY",
]

OTC_PAIRS = [
    "EURUSD-OTC",
    "GBPUSD-OTC",
    "USDJPY-OTC",
    "EURJPY-OTC",
    "GBPJPY-OTC",
    "NZDUSD-OTC",
]

TIMEFRAMES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
}

CANDLE_COUNT = 250

SCAN_INTERVAL = 30

MAX_SIGNALS_PER_DAY = 4

MIN_SCORE = 10
MIN_CONFIDENCE = 75

ACCOUNT_MODE = "PRACTICE"

AUTO_TRADE = False

STATE_FILE = "v42_state.json"


# ============================================================
# ENVIRONMENT
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
        print("Telegram credentials missing.", flush=True)
        return False

    try:

        url = (
            f"https://api.telegram.org/bot"
            f"{BOT_TOKEN}/sendMessage"
        )

        payload = {
            "chat_id": CHAT_ID,
            "text": message,
        }

        response = requests.post(
            url,
            json=payload,
            timeout=20,
        )

        if response.status_code == 200:

            print(
                "✅ Telegram message sent.",
                flush=True
            )

            return True

        print(
            f"Telegram error: {response.status_code}",
            flush=True
        )

        return False

    except Exception as e:

        print(
            f"Telegram exception: {e}",
            flush=True
        )

        return False


# ============================================================
# STATE
# ============================================================

def current_day():

    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")


def default_state():

    return {
        "date": current_day(),
        "signals_today": 0,
        "last_signals": {},
        "history": [],
    }


def load_state():

    try:

        if not os.path.exists(STATE_FILE):
            return default_state()

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            state = json.load(f)

        if state.get("date") != current_day():

            return default_state()

        return state

    except Exception:

        return default_state()


def save_state(state):

    try:

        with open(
            STATE_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                state,
                f,
                indent=2
            )

    except Exception as e:

        print(
            f"State save error: {e}",
            flush=True
        )


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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    result = 100 - (
        100 / (1 + rs)
    )

    return result.fillna(50)


def atr(df, period=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def macd(series):

    fast = ema(series, 12)
    slow = ema(series, 26)

    line = fast - slow

    signal = ema(line, 9)

    histogram = line - signal

    return line, signal, histogram


# ============================================================
# CANDLE DATA
# ============================================================

def get_candles(
    iq,
    pair,
    timeframe,
    count=CANDLE_COUNT
):

    try:

        candles = iq.get_candles(
            pair,
            timeframe,
            count,
            time.time()
        )

        if not candles:
            return None

        df = pd.DataFrame(candles)

        required = [
            "open",
            "close",
            "min",
            "max",
        ]

        for column in required:

            if column not in df.columns:
                return None

        df = df.rename(
            columns={
                "min": "low",
                "max": "high",
            }
        )

        df = df[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ].copy()

        for column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna()

        if len(df) < 80:
            return None

        return df.reset_index(drop=True)

    except Exception as e:

        print(
            f"Candle error {pair}: {e}",
            flush=True
        )

        return None


# ============================================================
# MARKET AVAILABILITY
# ============================================================

def get_open_assets(iq):

    try:

        return iq.get_all_open_time()

    except Exception as e:

        print(
            f"Open-time check failed: {e}",
            flush=True
        )

        return {}


def asset_is_open(open_assets, pair):

    try:

        for market_type in [
            "turbo",
            "binary",
            "digital",
        ]:

            market = open_assets.get(
                market_type,
                {}
            )

            asset = market.get(pair)

            if asset and asset.get("open") is True:

                return True

        return False

    except Exception:

        return False


def choose_available_asset(
    iq,
    regular_pair,
    open_assets
):

    # --------------------------------------------------------
    # FIRST: regular market
    # --------------------------------------------------------

    if asset_is_open(
        open_assets,
        regular_pair
    ):

        return regular_pair, "REGULAR"

    # --------------------------------------------------------
    # SECOND: OTC
    # --------------------------------------------------------

    otc_pair = (
        regular_pair
        + "-OTC"
    )

    if asset_is_open(
        open_assets,
        otc_pair
    ):

        return otc_pair, "OTC"

    # --------------------------------------------------------
    # NOTHING OPEN
    # --------------------------------------------------------

    return None, None


# ============================================================
# TREND / MARKET STRUCTURE
# ============================================================

def timeframe_bias(df):

    close = df["close"]

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e100 = ema(close, 100)

    last = close.iloc[-1]

    if (
        last > e20.iloc[-1]
        and e20.iloc[-1] > e50.iloc[-1]
        and e50.iloc[-1] > e100.iloc[-1]
    ):

        return "CALL"

    if (
        last < e20.iloc[-1]
        and e20.iloc[-1] < e50.iloc[-1]
        and e50.iloc[-1] < e100.iloc[-1]
    ):

        return "PUT"

    return "NEUTRAL"


def support_resistance(df):

    recent = df.tail(50)

    support = recent["low"].min()
    resistance = recent["high"].max()

    return support, resistance


# ============================================================
# ENTRY ANALYSIS
# ============================================================

def analyze_entry(df):

    close = df["close"]

    e9 = ema(close, 9)
    e21 = ema(close, 21)
    e50 = ema(close, 50)

    r = rsi(close)

    macd_line, macd_signal, macd_hist = macd(
        close
    )

    a = atr(df)

    score_call = 0
    score_put = 0

    reasons_call = []
    reasons_put = []

    # --------------------------------------------------------
    # EMA STRUCTURE
    # --------------------------------------------------------

    if (
        e9.iloc[-1]
        > e21.iloc[-1]
        > e50.iloc[-1]
    ):

        score_call += 3
        reasons_call.append(
            "EMA bullish alignment"
        )

    elif (
        e9.iloc[-1]
        < e21.iloc[-1]
        < e50.iloc[-1]
    ):

        score_put += 3
        reasons_put.append(
            "EMA bearish alignment"
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    current_rsi = float(
        r.iloc[-1]
    )

    if 52 <= current_rsi <= 68:

        score_call += 2
        reasons_call.append(
            f"RSI bullish ({current_rsi:.1f})"
        )

    elif 32 <= current_rsi <= 48:

        score_put += 2
        reasons_put.append(
            f"RSI bearish ({current_rsi:.1f})"
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if (
        macd_line.iloc[-1]
        > macd_signal.iloc[-1]
        and macd_hist.iloc[-1] > 0
    ):

        score_call += 2
        reasons_call.append(
            "MACD bullish momentum"
        )

    elif (
        macd_line.iloc[-1]
        < macd_signal.iloc[-1]
        and macd_hist.iloc[-1] < 0
    ):

        score_put += 2
        reasons_put.append(
            "MACD bearish momentum"
        )

    # --------------------------------------------------------
    # CANDLE MOMENTUM
    # --------------------------------------------------------

    last = df.iloc[-1]

    candle_body = abs(
        last["close"] - last["open"]
    )

    candle_range = (
        last["high"] - last["low"]
    )

    if candle_range > 0:

        body_ratio = (
            candle_body / candle_range
        )

        if body_ratio >= 0.55:

            if last["close"] > last["open"]:

                score_call += 2
                reasons_call.append(
                    "Strong bullish candle"
                )

            elif last["close"] < last["open"]:

                score_put += 2
                reasons_put.append(
                    "Strong bearish candle"
                )

    # --------------------------------------------------------
    # ATR / VOLATILITY
    # --------------------------------------------------------

    atr_value = float(
        a.iloc[-1]
    )

    if atr_value > 0:

        score_call += 1
        score_put += 1

    # --------------------------------------------------------
    # RECENT CANDLE AGREEMENT
    # --------------------------------------------------------

    previous = df.iloc[-2]

    if (
        last["close"] > last["open"]
        and previous["close"] > previous["open"]
    ):

        score_call += 1
        reasons_call.append(
            "Two-candle bullish momentum"
        )

    elif (
        last["close"] < last["open"]
        and previous["close"] < previous["open"]
    ):

        score_put += 1
        reasons_put.append(
            "Two-candle bearish momentum"
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if score_call > score_put:

        return {
            "direction": "CALL",
            "score": score_call,
            "rsi": current_rsi,
            "reasons": reasons_call,
        }

    if score_put > score_call:

        return {
            "direction": "PUT",
            "score": score_put,
            "rsi": current_rsi,
            "reasons": reasons_put,
        }

    return None


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    entry_score,
    bias_5m,
    bias_15m,
    direction
):

    confidence = 50

    # Entry strength

    confidence += (
        entry_score * 2
    )

    # Higher timeframe confirmation

    if bias_5m == direction:

        confidence += 8

    if bias_15m == direction:

        confidence += 10

    if (
        bias_5m == direction
        and bias_15m == direction
    ):

        confidence += 7

    # Contradiction penalty

    if (
        bias_5m != "NEUTRAL"
        and bias_5m != direction
    ):

        confidence -= 8

    if (
        bias_15m != "NEUTRAL"
        and bias_15m != direction
    ):

        confidence -= 10

    confidence = max(
        0,
        min(95, confidence)
    )

    return int(confidence)


# ============================================================
# ADAPTIVE EXPIRY
# ============================================================

def choose_expiry(
    confidence,
    bias_5m,
    bias_15m,
    entry_score
):

    if (
        confidence >= 88
        and bias_5m == bias_15m
        and bias_5m != "NEUTRAL"
        and entry_score >= 14
    ):

        return 3

    if (
        confidence >= 82
        and bias_5m != "NEUTRAL"
        and bias_5m == bias_15m
    ):

        return 5

    if (
        confidence >= 75
        and (
            bias_5m != "NEUTRAL"
            or bias_15m != "NEUTRAL"
        )
    ):

        return 5

    return 10


# ============================================================
# COMPLETE PAIR ANALYSIS
# ============================================================

def analyze_pair(
    iq,
    pair,
    market_type
):

    print(
        f"🔎 Analyzing {pair} [{market_type}]",
        flush=True
    )

    df_1m = get_candles(
        iq,
        pair,
        TIMEFRAMES["1m"]
    )

    df_5m = get_candles(
        iq,
        pair,
        TIMEFRAMES["5m"]
    )

    df_15m = get_candles(
        iq,
        pair,
        TIMEFRAMES["15m"]
    )

    if (
        df_1m is None
        or df_5m is None
        or df_15m is None
    ):

        print(
            f"{pair}: insufficient candles",
            flush=True
        )

        return None

    bias_5m = timeframe_bias(
        df_5m
    )

    bias_15m = timeframe_bias(
        df_15m
    )

    entry = analyze_entry(
        df_1m
    )

    if not entry:

        return None

    direction = entry[
        "direction"
    ]

    score = entry[
        "score"
    ]

    if score < MIN_SCORE:

        print(
            f"{pair}: score {score} rejected",
            flush=True
        )

        return None

    confidence = calculate_confidence(
        score,
        bias_5m,
        bias_15m,
        direction
    )

    if confidence < MIN_CONFIDENCE:

        print(
            f"{pair}: confidence "
            f"{confidence}% rejected",
            flush=True
        )

        return None

    # --------------------------------------------------------
    # Separation / confirmation
    # --------------------------------------------------------

    separation = 0

    if bias_5m == direction:

        separation += 1

    if bias_15m == direction:

        separation += 1

    if separation < 1:

        return None

    expiry = choose_expiry(
        confidence,
        bias_5m,
        bias_15m,
        score
    )

    support, resistance = (
        support_resistance(df_1m)
    )

    entry_price = float(
        df_1m["close"].iloc[-1]
    )

    return {
        "pair": pair,
        "market": market_type,
        "direction": direction,
        "score": score,
        "confidence": confidence,
        "expiry": expiry,
        "rsi": entry["rsi"],
        "entry": entry_price,
        "bias_5m": bias_5m,
        "bias_15m": bias_15m,
        "separation": separation,
        "support": float(support),
        "resistance": float(resistance),
        "reasons": entry["reasons"],
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ============================================================
# SIGNAL FORMAT
# ============================================================

def format_signal(result):

    direction_icon = (
        "🟢"
        if result["direction"] == "CALL"
        else "🔴"
    )

    reasons = "\n".join(
        "• " + reason
        for reason in result["reasons"]
    )

    return (
        "🎯 V4.2 PREMIUM SIGNAL\n\n"

        f"💱 Pair: {result['pair']}\n"
        f"📊 Market: {result['market']}\n"

        f"{direction_icon} Direction: "
        f"{result['direction']}\n"

        f"⏱ Expiry: "
        f"{result['expiry']} minutes\n"

        f"📊 Score: "
        f"{result['score']}\n"

        f"🧠 Model Confidence: "
        f"{result['confidence']}%\n"

        f"💰 Entry reference: "
        f"{result['entry']:.6f}\n"

        f"📈 RSI: "
        f"{result['rsi']:.1f}\n\n"

        f"5M Bias: {result['bias_5m']}\n"
        f"15M Bias: {result['bias_15m']}\n\n"

        "Why:\n"
        f"{reasons}\n\n"

        "⚠️ Practice/demo signal\n"
        "🤖 Auto-trading: OFF\n"
        "🔢 Daily limit: 4 signals maximum"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "🔥 V4.2 PROFESSIONAL SIGNAL ENGINE",
        flush=True
    )

    print(
        "🛡️ Maximum 4 signals per day",
        flush=True
    )

    print(
        "🌐 Regular + OTC availability detection enabled",
        flush=True
    )

    print(
        "🤖 Auto-trading: OFF",
        flush=True
    )

    state = load_state()

    telegram(
        "🟢 V4.2 SCANNER STARTING\n\n"
        "Professional signal engine online.\n"
        "Maximum 4 signals per day.\n"
        "Regular + OTC market detection enabled.\n"
        "Auto-trading: OFF."
    )

    print(
        "Connecting to IQ Option...",
        flush=True
    )

    iq = IQ_Option(
        IQ_EMAIL,
        IQ_PASSWORD
    )

    iq.connect()

    if not iq.check_connect():

        print(
            "❌ IQ Option connection failed.",
            flush=True
        )

        telegram(
            "🔴 V4.2 ERROR\n\n"
            "IQ Option connection failed."
        )

        return

    print(
        "✅ IQ Option connected",
        flush=True
    )

    try:

        iq.change_balance(
            ACCOUNT_MODE
        )

    except Exception:

        pass

    telegram(
        "🟢 V4.2 SCANNER ONLINE\n\n"
        "Market-status protection: ON\n"
        "OTC support: ON\n"
        "Daily maximum: 4\n"
        "Auto-trading: OFF"
    )

    while True:

        try:

            # ------------------------------------------------
            # RESET DAILY STATE
            # ------------------------------------------------

            if state["date"] != current_day():

                state = default_state()

                save_state(state)

                print(
                    "🔄 New trading day.",
                    flush=True
                )

            # ------------------------------------------------
            # HARD DAILY LIMIT
            # ------------------------------------------------

                        if (
                state["signals_today"]
                >= MAX_SIGNALS_PER_DAY
            ):

                print(
                    "🛑 Daily limit reached: "
                    f"{MAX_SIGNALS_PER_DAY}/"
                    f"{MAX_SIGNALS_PER_DAY}",
                    flush=True
                )

                time.sleep(300)

                continue

            # ------------------------------------------------
            # GET REAL MARKET STATUS
            # ------------------------------------------------

            open_assets = get_open_assets(iq)

            if not open_assets:

                print(
                    "⚠️ Could not obtain market status.",
                    flush=True
                )

                time.sleep(60)

                continue

            candidates = []

            # ------------------------------------------------
            # SCAN PAIRS
            # ------------------------------------------------

            for regular_pair in PAIRS:

                asset, market_type = (
                    choose_available_asset(
                        iq,
                        regular_pair,
                        open_assets
                    )
                )

                if not asset:

                    print(
                        f"⏸ {regular_pair}: "
                        "regular + OTC unavailable",
                        flush=True
                    )

                    continue

                print(
                    f"✅ {asset} is OPEN "
                    f"[{market_type}]",
                    flush=True
                )

                result = analyze_pair(
                    iq,
                    asset,
                    market_type
                )

                if result:

                    candidates.append(result)

            # ------------------------------------------------
            # SELECT BEST SIGNAL
            # ------------------------------------------------

            if candidates:

                candidates.sort(
                    key=lambda x: (
                        x["confidence"],
                        x["score"],
                        x["separation"],
                    ),
                    reverse=True
                )

                best = candidates[0]

                last = state[
                    "last_signals"
                ].get(
                    best["pair"]
                )

                duplicate = False

                if last:

                    if (
                        last["direction"]
                        == best["direction"]
                    ):

                        duplicate = True

                if duplicate:

                    print(
                        f"⏸ {best['pair']}: "
                        "duplicate direction ignored",
                        flush=True
                    )

                else:

                    message = format_signal(best)

                    if telegram(message):

                        state[
                            "signals_today"
                        ] += 1

                        state[
                            "last_signals"
                        ][
                            best["pair"]
                        ] = {
                            "direction":
                                best["direction"],
                            "time":
                                datetime.now(
                                    timezone.utc
                                ).isoformat(),
                        }

                        state[
                            "history"
                        ].append(best)

                        save_state(state)

                        print(
                            "🚨 SIGNAL SENT",
                            flush=True
                        )

                        print(
                            f"{best['pair']} "
                            f"{best['direction']} "
                            f"{best['confidence']}%",
                            flush=True
                        )

            else:

                print(
                    "No qualifying setup.",
                    flush=True
                )

            time.sleep(
                SCAN_INTERVAL
            )

        except KeyboardInterrupt:

            print(
                "Scanner stopped.",
                flush=True
            )

            break

        except Exception as e:

            print(
                f"⚠️ Scanner error: {e}",
                flush=True
            )

            traceback.print_exc()

            time.sleep(30)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
    
