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
# V4 CONFIG
# ============================================================

PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "EURJPY",
    "GBPJPY",
    "NZDJPY",
]

CANDLE_TIMEFRAME = 60          # 1-minute candles
CANDLE_COUNT = 300

SCAN_INTERVAL = 30             # seconds
MAX_SIGNALS_PER_DAY = 4

MIN_SCORE = 8
MIN_CONFIDENCE = 72

# Automatic trading is deliberately disabled.
AUTO_TRADE = False

# Practice/demo account.
ACCOUNT_MODE = "PRACTICE"

STATE_FILE = "v4_state.json"


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
        print("❌ Telegram credentials missing", flush=True)
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
            f"Telegram HTTP {response.status_code}: {response.text}",
            flush=True
        )

        return response.status_code == 200

    except Exception as e:
        print(f"❌ Telegram error: {e}", flush=True)
        return False


# ============================================================
# STATE
# ============================================================

def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_state():
    default = {
        "date": today(),
        "signals_today": 0,
        "last_signal": {},
        "history": [],
    }

    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                state = json.load(f)

            if state.get("date") != today():
                return default

            return state

    except Exception as e:
        print(f"State load warning: {e}", flush=True)

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
    return series.ewm(span=period, adjust=False).mean()


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
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


def macd(series):
    fast = ema(series, 12)
    slow = ema(series, 26)

    line = fast - slow
    signal = ema(line, 9)

    return line, signal


# ============================================================
# CANDLE DATA
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
            return None

        df = pd.DataFrame(candles)

        required = ["open", "close", "min", "max"]

        for column in required:
            if column not in df.columns:
                print(
                    f"{pair}: missing candle column {column}",
                    flush=True
                )
                return None

        df = df.rename(
            columns={
                "min": "low",
                "max": "high",
            }
        )

        df = df[
            ["open", "high", "low", "close"]
        ].copy()

        for column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna()

        if len(df) < 100:
            return None

        return df.reset_index(drop=True)

    except Exception as e:
        print(
            f"{pair}: candle error: {e}",
            flush=True
        )
        return None


# ============================================================
# MARKET ANALYSIS
# ============================================================

def analyze(pair, df):
    if df is None or len(df) < 100:
        return None

    close = df["close"]

    df["ema9"] = ema(close, 9)
    df["ema21"] = ema(close, 21)
    df["ema50"] = ema(close, 50)
    df["rsi"] = rsi(close)

    df["atr"] = atr(df)

    macd_line, macd_signal = macd(close)

    df["macd"] = macd_line
    df["macd_signal"] = macd_signal

    current = df.iloc[-1]
    previous = df.iloc[-2]

    score_call = 0
    score_put = 0

    reasons_call = []
    reasons_put = []

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if (
        current["ema9"] > current["ema21"]
        and current["ema21"] > current["ema50"]
    ):
        score_call += 3
        reasons_call.append("EMA bullish alignment")

    if (
        current["ema9"] < current["ema21"]
        and current["ema21"] < current["ema50"]
    ):
        score_put += 3
        reasons_put.append("EMA bearish alignment")

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if (
        current["macd"] > current["macd_signal"]
        and previous["macd"] <= previous["macd_signal"]
    ):
        score_call += 2
        reasons_call.append("MACD bullish cross")

    elif current["macd"] > current["macd_signal"]:
        score_call += 1
        reasons_call.append("MACD bullish")

    if (
        current["macd"] < current["macd_signal"]
        and previous["macd"] >= previous["macd_signal"]
    ):
        score_put += 2
        reasons_put.append("MACD bearish cross")

    elif current["macd"] < current["macd_signal"]:
        score_put += 1
        reasons_put.append("MACD bearish")

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    r = float(current["rsi"])

    if 52 <= r <= 68:
        score_call += 2
        reasons_call.append(f"RSI bullish zone ({r:.1f})")

    if 32 <= r <= 48:
        score_put += 2
        reasons_put.append(f"RSI bearish zone ({r:.1f})")

    # Avoid chasing extreme RSI.
    if r > 75:
        score_call -= 2

    if r < 25:
        score_put -= 2

    # --------------------------------------------------------
    # CANDLE MOMENTUM
    # --------------------------------------------------------

    candle_range = current["high"] - current["low"]

    if candle_range > 0:

        body = abs(
            current["close"] - current["open"]
        )

        body_ratio = body / candle_range

        if body_ratio >= 0.55:

            if current["close"] > current["open"]:
                score_call += 2
                reasons_call.append("Strong bullish candle")

            elif current["close"] < current["open"]:
                score_put += 2
                reasons_put.append("Strong bearish candle")

    # --------------------------------------------------------
    # SHORT-TERM MOMENTUM
    # --------------------------------------------------------

    if current["close"] > previous["close"]:
        score_call += 1

    if current["close"] < previous["close"]:
        score_put += 1

    # --------------------------------------------------------
    # DETERMINE DIRECTION
    # --------------------------------------------------------

    if score_call > score_put:
        direction = "CALL"
        score = score_call
        reasons = reasons_call

    elif score_put > score_call:
        direction = "PUT"
        score = score_put
        reasons = reasons_put

    else:
        return None

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    separation = abs(score_call - score_put)

    confidence = min(
        95,
        55
        + score * 3
        + separation * 3
    )

    # --------------------------------------------------------
    # ADAPTIVE EXPIRY
    # --------------------------------------------------------

    if score >= 11 and confidence >= 82:
        expiry = 3

    elif score >= 9 and confidence >= 76:
        expiry = 5

    else:
        expiry = 5

    # --------------------------------------------------------
    # REJECT WEAK SETUPS
    # --------------------------------------------------------

    if score < MIN_SCORE:
        return None

    if confidence < MIN_CONFIDENCE:
        return None

    if separation < 2:
        return None

    return {
        "pair": pair,
        "direction": direction,
        "score": score,
        "confidence": round(confidence),
        "expiry": expiry,
        "price": float(current["close"]),
        "rsi": round(r, 1),
        "reasons": reasons,
    }


# ============================================================
# CONNECTION
# ============================================================

def connect_iq():
    print("Connecting to IQ Option...", flush=True)

    iq = IQ_Option(
        IQ_EMAIL,
        IQ_PASSWORD
    )

    success, reason = iq.connect()

    if not success:
        print(
            f"❌ IQ Option connection failed: {reason}",
            flush=True
        )

        telegram(
            "🔴 V4 ERROR\n\n"
            f"IQ Option connection failed.\n"
            f"Reason: {reason}"
        )

        return None

    print("✅ IQ Option connected", flush=True)

    try:
        iq.change_balance("PRACTICE")
    except Exception:
        pass

    return iq


# ============================================================
# TELEGRAM FORMATTING
# ============================================================

def signal_message(signal):
    direction_icon = (
        "🟢" if signal["direction"] == "CALL"
        else "🔴"
    )

    reasons = "\n".join(
        f"• {x}"
        for x in signal["reasons"][:5]
    )

    return (
        f"🎯 V4 PREMIUM SIGNAL\n\n"
        f"💱 Pair: {signal['pair']}\n"
        f"{direction_icon} Direction: {signal['direction']}\n"
        f"⏱ Expiry: {signal['expiry']} minutes\n"
        f"📊 Score: {signal['score']}\n"
        f"🧠 Confidence: {signal['confidence']}%\n"
        f"💰 Entry reference: {signal['price']}\n"
        f"📈 RSI: {signal['rsi']}\n\n"
        f"Why:\n"
        f"{reasons}\n\n"
        f"⚠️ Practice/demo signal\n"
        f"🤖 Auto-trading: OFF"
    )


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    print("=" * 60, flush=True)
    print("🚀 V4 PROFESSIONAL SIGNAL SCANNER", flush=True)
    print("=" * 60, flush=True)

    telegram(
        "🟢 V4 SCANNER STARTING\n\n"
        "Telegram: CONNECTED\n"
        "Market engine: STARTING\n"
        "Account: PRACTICE / DEMO\n"
        "Auto-trading: OFF\n\n"
        "Maximum signals today: 4"
    )

    if not IQ_EMAIL or not IQ_PASSWORD:
        telegram(
            "🔴 V4 ERROR\n\n"
            "IQ Option credentials are missing."
        )
        return

    iq = connect_iq()

    if iq is None:
        return

    state = load_state()

    telegram(
        "🟢 V4 SCANNER ONLINE\n\n"
        "Market connection: OK\n"
        "Scanning selected pairs...\n"
        "Waiting for high-quality setups."
    )

    print("Scanner is running...", flush=True)

    while True:

        try:

            # Reset daily counter if date changed.
            if state["date"] != today():
                state = load_state()

            if state["signals_today"] >= MAX_SIGNALS_PER_DAY:

                print(
                    "Daily signal limit reached. "
                    "Waiting for tomorrow.",
                    flush=True
                )

                time.sleep(300)
                continue

            candidates = []

            for pair in PAIRS:

                print(
                    f"Analyzing {pair}...",
                    flush=True
                )

                df = get_candles(
                    iq,
                    pair,
                    CANDLE_TIMEFRAME,
                    CANDLE_COUNT
                )

                if df is None:
                    continue

                signal = analyze(
                    pair,
                    df
                )

                if signal:
                    candidates.append(signal)

            if candidates:

                # Strongest setup first.
                candidates.sort(
                    key=lambda x: (
                        x["confidence"],
                        x["score"]
                    ),
                    reverse=True
                )

                best = candidates[0]

                last_signal = state["last_signal"]

                # Avoid repeating same pair/direction immediately.
                if (
                    last_signal.get("pair") == best["pair"]
                    and last_signal.get("direction")
                    == best["direction"]
                ):
                    print(
                        "Duplicate setup ignored.",
                        flush=True
                    )

                else:

                    message = signal_message(best)

                    if telegram(message):

                        state["signals_today"] += 1

                        state["last_signal"] = {
                            "pair": best["pair"],
                            "direction": best["direction"],
                            "time": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }

                        state["history"].append(best)

                        save_state(state)

                        print(
                            f"✅ SIGNAL SENT: "
                            f"{best['pair']} "
                            f"{best['direction']} "
                            f"{best['confidence']}%",
                            flush=True
                        )

                        # Do NOT automatically trade.
                        if AUTO_TRADE:
                            print(
                                "AUTO_TRADE requested, "
                                "but this build does not "
                                "place trades.",
                                flush=True
                            )

            else:

                print(
                    "No high-quality setup found.",
                    flush=True
                )

            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:

            print(
                "Scanner stopped.",
                flush=True
            )

            break

        except Exception as e:

            print(
                "⚠️ Scanner error:",
                str(e),
                flush=True
            )

            traceback.print_exc()

            telegram(
                "⚠️ V4 SCANNER WARNING\n\n"
                f"{str(e)[:500]}"
            )

            time.sleep(30)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
