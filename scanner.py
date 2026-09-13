# VETRA-X LEARNING SIGNAL — V6.1
# IQ Option PRACTICE / SIGNAL-ONLY scanner
# 1-minute expiry. No real-money trading. No order calls.

import os
import json
import time
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from iqoptionapi.stable_api import IQ_Option


MODE = "PRACTICE"
AUTO_TRADE = False

MAX_DAILY_SIGNALS = 4
MIN_CONFIDENCE = 0.78
EXPIRY_SECONDS = 60
MIN_ENTRY_LEAD_SECONDS = 15

# V6.1: no 15-second candle-age requirement.
# We allow delayed GitHub Actions runs, but reject genuinely stale data.
MAX_CLOSED_CANDLE_AGE_SECONDS = 180
LEARNING_MIN_EXAMPLES = 40
STATE_FILE = Path("vetrax_learning_state.json")

IQ_EMAIL = os.getenv("IQ_EMAIL", "")
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

REGULAR_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD",
    "NZDUSD","EURJPY","GBPJPY","EURGBP","EURCAD","AUDJPY",
    "AUDCHF","CADCHF","CADJPY","CHFJPY","EURAUD","EURNZD",
    "GBPAUD","GBPCAD","GBPCHF","GBPNZD"
]

OTC_PAIRS = [
    "EURUSD-OTC","GBPUSD-OTC","USDJPY-OTC","USDCHF-OTC",
    "AUDUSD-OTC","USDCAD-OTC","NZDUSD-OTC","EURJPY-OTC",
    "GBPJPY-OTC","EURGBP-OTC","EURCAD-OTC","AUDJPY-OTC",
    "GBPCHF-OTC","NZDJPY-OTC","NZDCAD-OTC","AUDNZD-OTC",
    "CADCHF-OTC","CADJPY-OTC","CHFJPY-OTC"
]

REGULAR_SET = set(REGULAR_PAIRS)
OTC_SET = set(OTC_PAIRS)

PREFERRED_ORDER = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","EURJPY","GBPJPY",
    "EURGBP","AUDUSD","USDCAD","NZDUSD","EURCAD","AUDJPY",
    "GBPCHF","CADJPY","CHFJPY","AUDCHF","CADCHF","EURAUD",
    "EURNZD","GBPAUD","GBPCAD","GBPNZD"
]

FEATURE_NAMES = [
    "ret1","ret2","ret3","ret5","body","range","upper_wick",
    "lower_wick","ema9_gap","ema21_gap","ema50_gap","rsi",
    "macd","macd_signal","macd_hist","atr_pct","trend5",
    "momentum5","break_high","break_low","volume_ratio",
    "hour_sin","hour_cos"
]

CURRENT_STATE = None


def today_lagos():
    return (datetime.now(timezone.utc) + timedelta(hours=1)).date().isoformat()


def safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials not configured.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=15
        )
        if r.ok:
            return True
        print("Telegram error:", r.status_code, r.text[:300])
    except Exception as e:
        print("Telegram exception:", repr(e))
    return False


def default_state():
    return {
        "date": today_lagos(),
        "signals_today": 0,
        "examples": [],
        "pending": [],
        "wins": 0,
        "losses": 0
    }


def load_state():
    if not STATE_FILE.exists():
        return default_state()
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            s = json.load(f)
        if not isinstance(s, dict):
            return default_state()
        if s.get("date") != today_lagos():
            s["date"] = today_lagos()
            s["signals_today"] = 0
        s.setdefault("examples", [])
        s.setdefault("pending", [])
        s.setdefault("wins", 0)
        s.setdefault("losses", 0)
        s.setdefault("signals_today", 0)
        return s
    except Exception as e:
        print("Could not load learning state:", repr(e))
        return default_state()


def save_state(state):
    try:
        tmp = STATE_FILE.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f, separators=(",", ":"))
        tmp.replace(STATE_FILE)
    except Exception as e:
        print("Could not save learning state:", repr(e))


def load_binary_status(iq):
    regular_open = set()
    otc_open = set()
    try:
        result = iq.get_all_init()
        if not isinstance(result, dict):
            print("Broker status response was not a dictionary.")
            return regular_open, otc_open

        for mode in ("turbo", "binary"):
            section = result.get(mode, {})
            actives = section.get("actives", {}) if isinstance(section, dict) else {}
            if not isinstance(actives, dict):
                continue

            for key, detail in actives.items():
                name = None
                if isinstance(detail, dict):
                    name = detail.get("name") or detail.get("active_name") or detail.get("symbol")
                name = name or str(key)
                if "." in name:
                    name = name.split(".")[-1]

                enabled = bool(detail.get("enabled", True)) if isinstance(detail, dict) else True
                suspended = bool(detail.get("suspended", False)) if isinstance(detail, dict) else False
                if not enabled or suspended:
                    continue

                if name in REGULAR_SET:
                    regular_open.add(name)
                elif name in OTC_SET:
                    otc_open.add(name)

        print(
            f"Broker binary/turbo status: {len(regular_open)} regular open, "
            f"{len(otc_open)} OTC open."
        )
        print(
            "Regular supported open:",
            ", ".join(sorted(regular_open)) if regular_open else "NONE"
        )
        print(
            "OTC supported open:",
            ", ".join(sorted(otc_open)) if otc_open else "NONE"
        )
    except Exception as e:
        print("Broker status error:", repr(e))
    return regular_open, otc_open


def get_candles(iq, pair, count=120):
    try:
        candles = iq.get_candles(pair, 60, count, time.time())
        if not candles:
            return None

        rows = []
        for c in candles:
            rows.append({
                "ts": int(c.get("from", 0)),
                "open": safe_float(c.get("open")),
                "close": safe_float(c.get("close")),
                "high": safe_float(c.get("max")),
                "low": safe_float(c.get("min")),
                "volume": safe_float(c.get("volume"), 1.0)
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return None

        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        for col in ["open","close","high","low","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["ts","open","close","high","low"]).reset_index(drop=True)

        return df if len(df) >= 60 else None
    except Exception as e:
        print(f"get_candles {pair} error:", repr(e))
        return None


def latest_closed_candles(df):
    if df is None or df.empty:
        return None
    now = time.time()
    closed = df[df["ts"] + 60 <= now].copy()
    return closed.reset_index(drop=True) if len(closed) >= 60 else None


def candle_data_is_current(closed):
    if closed is None or closed.empty:
        return False
    closed_at = safe_float(closed.iloc[-1]["ts"]) + 60
    return (time.time() - closed_at) <= MAX_CLOSED_CANDLE_AGE_SECONDS


def next_entry_time(closed):
    return int(closed.iloc[-1]["ts"]) + 60


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(series):
    fast = ema(series, 12)
    slow = ema(series, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal, line - signal


def atr(df, period=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def feature_row(df):
    if df is None or len(df) < 60:
        return None

    x = df.copy()
    x["ema9"] = ema(x["close"], 9)
    x["ema21"] = ema(x["close"], 21)
    x["ema50"] = ema(x["close"], 50)
    x["rsi"] = rsi(x["close"])
    ml, ms, mh = macd(x["close"])
    x["macd"], x["macd_signal"], x["macd_hist"] = ml, ms, mh
    x["atr"] = atr(x)

    x["ret1"] = x["close"].pct_change(1)
    x["ret2"] = x["close"].pct_change(2)
    x["ret3"] = x["close"].pct_change(3)
    x["ret5"] = x["close"].pct_change(5)
    x["body"] = (x["close"] - x["open"]) / x["open"].replace(0, np.nan)
    x["range"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)

    top = x[["open","close"]].max(axis=1)
    bottom = x[["open","close"]].min(axis=1)
    x["upper_wick"] = (x["high"] - top) / x["close"].replace(0, np.nan)
    x["lower_wick"] = (bottom - x["low"]) / x["close"].replace(0, np.nan)

    x["ema9_gap"] = (x["close"] - x["ema9"]) / x["close"].replace(0, np.nan)
    x["ema21_gap"] = (x["close"] - x["ema21"]) / x["close"].replace(0, np.nan)
    x["ema50_gap"] = (x["close"] - x["ema50"]) / x["close"].replace(0, np.nan)
    x["atr_pct"] = x["atr"] / x["close"].replace(0, np.nan)
    x["trend5"] = x["close"] / x["close"].shift(5) - 1
    x["momentum5"] = (x["close"] - x["close"].shift(5)) / x["atr"].replace(0, np.nan)

    hi = x["high"].shift(1).rolling(20).max()
    lo = x["low"].shift(1).rolling(20).min()
    x["break_high"] = (x["close"] - hi) / x["close"].replace(0, np.nan)
    x["break_low"] = (lo - x["close"]) / x["close"].replace(0, np.nan)

    vol = x["volume"].rolling(20).mean()
    x["volume_ratio"] = x["volume"] / vol.replace(0, np.nan)

    hours = pd.to_datetime(x["ts"], unit="s", utc=True).dt.hour.astype(float)
    x["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    row = x.iloc[-1]
    arr = np.asarray([safe_float(row.get(n)) for n in FEATURE_NAMES], dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def rule_setup(df):
    if df is None or len(df) < 60:
        return None

    close = float(df.iloc[-1]["close"])
    prev = float(df.iloc[-2]["close"])

    e9 = float(ema(df["close"], 9).iloc[-1])
    e21 = float(ema(df["close"], 21).iloc[-1])
    e50 = float(ema(df["close"], 50).iloc[-1])
    rv = float(rsi(df["close"]).iloc[-1])
    ml, ms, mh = macd(df["close"])
    macd_now = float(ml.iloc[-1])
    signal_now = float(ms.iloc[-1])
    hist_now = float(mh.iloc[-1])

    candle = df.iloc[-1]
    bullish = candle["close"] > candle["open"]
    bearish = candle["close"] < candle["open"]

    recent_high = float(df["high"].iloc[-21:-1].max())
    recent_low = float(df["low"].iloc[-21:-1].min())

    buy = sell = 0.0

    if e9 > e21: buy += 16
    elif e9 < e21: sell += 16
    if e21 > e50: buy += 12
    elif e21 < e50: sell += 12
    if close > e9: buy += 10
    elif close < e9: sell += 10

    if 50 <= rv <= 68: buy += 10
    elif 32 <= rv < 50: sell += 10
    elif rv > 72: sell += 5
    elif rv < 28: buy += 5

    if macd_now > signal_now and hist_now > 0: buy += 12
    elif macd_now < signal_now and hist_now < 0: sell += 12

    if bullish: buy += 8
    elif bearish: sell += 8
    if close > prev: buy += 5
    elif close < prev: sell += 5
    if close > recent_high: buy += 14
    if close < recent_low: sell += 14

    body = abs(float(candle["close"]) - float(candle["open"]))
    rng = max(float(candle["high"]) - float(candle["low"]), 1e-12)
    if body / rng >= 0.55:
        if bullish: buy += 5
        elif bearish: sell += 5

    best = max(buy, sell)
    if best <= 0:
        return None

    direction = "BUY" if buy >= sell else "SELL"
    confidence = min(0.97, max(0.50, best / 87.0))

    return {
        "direction": direction,
        "buy_score": round(buy, 2),
        "sell_score": round(sell, 2),
        "confidence": confidence
    }


def train_predict(state, current_x):
    examples = state.get("examples", [])
    if len(examples) < LEARNING_MIN_EXAMPLES:
        return None

    X, y = [], []
    for ex in examples:
        try:
            f = np.asarray(ex["x"], dtype=float)
            label = int(ex["y"])
            if len(f) != len(FEATURE_NAMES) or label not in (0, 1):
                continue
            f[~np.isfinite(f)] = 0.0
            X.append(f)
            y.append(label)
        except Exception:
            continue

    if len(X) < LEARNING_MIN_EXAMPLES or len(set(y)) < 2:
        return None

    try:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(np.asarray(X))
        model = SGDClassifier(
            loss="log_loss",
            alpha=0.001,
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
        model.fit(Xs, np.asarray(y))
        cur = np.asarray(current_x, dtype=float).reshape(1, -1)
        cur[~np.isfinite(cur)] = 0.0
        return float(model.predict_proba(scaler.transform(cur))[0][1])
    except Exception as e:
        print("ML error:", repr(e))
        return None


def resolve_pending(iq, state):
    remaining = []
    for item in state.get("pending", []):
        try:
            if time.time() < float(item["resolve_ts"]):
                remaining.append(item)
                continue

            pair = item["pair"]
            direction = item["direction"]
            entry = float(item["entry_price"])
            df = get_candles(iq, pair, 15)

            if df is None:
                remaining.append(item)
                continue

            eligible = df[df["ts"] >= float(item["resolve_ts"])]
            if eligible.empty:
                remaining.append(item)
                continue

            exit_price = float(eligible.iloc[0]["close"])
            win = exit_price > entry if direction == "BUY" else exit_price < entry
            label = 1 if win else 0

            state["examples"].append({
                "x": [safe_float(v) for v in item["x"]],
                "y": label,
                "pair": pair,
                "direction": direction,
                "time": int(time.time())
            })
            state["examples"] = state["examples"][-3000:]

            if win:
                state["wins"] = int(state.get("wins", 0)) + 1
                result = "WIN"
            else:
                state["losses"] = int(state.get("losses", 0)) + 1
                result = "LOSS"

            print(
                f"Result: {pair} {direction} "
                f"entry={entry} exit={exit_price} -> {result}"
            )

            if item.get("sent", False):
                total = state["wins"] + state["losses"]
                acc = state["wins"] / total * 100 if total else 0
                send_telegram(
                    f"VETRA-X RESULT\n"
                    f"{pair} {direction}\n"
                    f"Result: {result}\n"
                    f"Learning examples: {len(state['examples'])}\n"
                    f"Learning record: {state['wins']}W / {state['losses']}L ({acc:.1f}%)\n"
                    f"Practice / signal only"
                )
        except Exception as e:
            print("Pending result error:", repr(e))
            remaining.append(item)

    state["pending"] = remaining[-1000:]


def ordered_pairs(regular_open, otc_open):
    pairs = []
    for p in PREFERRED_ORDER:
        if p in regular_open:
            pairs.append(p)
    for p in sorted(regular_open):
        if p not in pairs:
            pairs.append(p)
    for base in PREFERRED_ORDER:
        otc = base + "-OTC"
        if otc in otc_open:
            pairs.append(otc)
    for p in sorted(otc_open):
        if p not in pairs:
            pairs.append(p)
    return pairs


def scan_pair(iq, pair, state):
    df = get_candles(iq, pair, 120)
    closed = latest_closed_candles(df)

    if closed is None or not candle_data_is_current(closed):
        return None

    entry_ts = next_entry_time(closed)
    lead = entry_ts - time.time()

    # Critical V6.1 protection: never issue an already-started entry.
    if lead < MIN_ENTRY_LEAD_SECONDS:
        return None

    x = feature_row(closed)
    setup = rule_setup(closed)

    if x is None or setup is None:
        return None

    ml_probability = train_predict(state, x)
    combined = setup["confidence"]

    if ml_probability is not None:
        ml_conf = max(ml_probability, 1 - ml_probability)
        ml_direction = "BUY" if ml_probability >= 0.50 else "SELL"

        if ml_direction == setup["direction"]:
            combined = min(0.97, setup["confidence"] * 0.55 + ml_conf * 0.45)
        else:
            combined = min(0.74, setup["confidence"] * 0.65)

    return {
        "pair": pair,
        "direction": setup["direction"],
        "confidence": combined,
        "ml_probability": ml_probability,
        "entry_ts": int(entry_ts),
        "entry_price": float(closed.iloc[-1]["close"]),
        "x": x.tolist(),
        "lead": lead
    }


def main():
    global CURRENT_STATE

    state = load_state()
    CURRENT_STATE = state

    iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)

    try:
        iq.connect()
    except Exception as e:
        print("Connection error:", repr(e))
        return

    if not iq.check_connect():
        print("Could not connect to IQ Option.")
        return

    try:
        iq.change_balance(MODE)
    except Exception as e:
        print("Balance mode warning:", repr(e))

    print(f"Connected to IQ Option in {MODE} mode.")

    resolve_pending(iq, state)
    save_state(state)

    if int(state.get("signals_today", 0)) >= MAX_DAILY_SIGNALS:
        print(
            f"Daily signal limit reached: "
            f"{state.get('signals_today', 0)}/{MAX_DAILY_SIGNALS}"
        )
        return

    regular_open, otc_open = load_binary_status(iq)
    pairs = ordered_pairs(regular_open, otc_open)

    if not pairs:
        print("No supported broker-reported Forex pair is open.")
        return

    candidates = []
    for pair in pairs:
        try:
            result = scan_pair(iq, pair, state)
            if result is not None:
                candidates.append(result)
        except Exception as e:
            print(f"Scan error {pair}:", repr(e))

    if not candidates:
        print(
            "No supported open pair has usable current candles "
            "with a future entry. No signal."
        )
        save_state(state)
        return

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    best = candidates[0]

    if best["confidence"] < MIN_CONFIDENCE:
        print(
            f"Best setup below threshold: "
            f"{best['confidence'] * 100:.1f}%"
        )
        save_state(state)
        return

    entry_dt = datetime.fromtimestamp(best["entry_ts"], timezone.utc) + timedelta(hours=1)
    entry_string = entry_dt.strftime("%I:%M:%S %p")

    pair_label = best["pair"]
    market_label = "OTC" if pair_label.endswith("-OTC") else "REGULAR"

    if best["ml_probability"] is None:
        model_line = "learning warm-up"
    else:
        model_line = f"ML {best['ml_probability'] * 100:.1f}%"

    print("\n🧠 VETRA-X LEARNING SIGNAL")
    print(f"PAIR: {pair_label} ({market_label})")
    print(f"DIRECTION: {best['direction']}")
    print(f"CONFIDENCE: {best['confidence'] * 100:.1f}%")
    print(f"ENTRY: {entry_string} WAT")
    print(f"ENTRY LEAD: {best['lead']:.1f}s")
    print("EXPIRY: 1 minute")
    print(f"MODEL: {model_line}")
    print(f"LEARNING EXAMPLES: {len(state['examples'])}")
    print(f"TODAY: {state.get('signals_today', 0)}/{MAX_DAILY_SIGNALS}")

    message = (
        "🧠 VETRA-X LEARNING SIGNAL\n\n"
        f"PAIR: {pair_label} ({market_label})\n"
        f"DIRECTION: {best['direction']}\n"
        f"CONFIDENCE: {best['confidence'] * 100:.1f}%\n"
        f"ENTRY: {entry_string} WAT\n"
        "EXPIRY: 1 minute\n"
        f"MODEL: {model_line}\n"
        f"LEARNING EXAMPLES: {len(state['examples'])}\n"
        f"TODAY: {state.get('signals_today', 0) + 1}/{MAX_DAILY_SIGNALS}\n\n"
        "PRACTICE / SIGNAL ONLY\n"
        "No automatic trade."
    )

    sent = send_telegram(message)

    state["signals_today"] = int(state.get("signals_today", 0)) + 1
    state["pending"].append({
        "pair": best["pair"],
        "direction": best["direction"],
        "entry_price": best["entry_price"],
        "entry_ts": best["entry_ts"],
        "resolve_ts": best["entry_ts"] + EXPIRY_SECONDS,
        "x": best["x"],
        "sent": bool(sent),
        "created_ts": int(time.time())
    })
    state["pending"] = state["pending"][-1000:]

    save_state(state)
    print("Signal saved.")


if __name__ == "__main__":
    main()
