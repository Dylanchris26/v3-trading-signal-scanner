# VETRA-X LEARNING SIGNAL SCANNER
# PRACTICE / SIGNAL ONLY — NO AUTOMATIC TRADING
#
# This version:
# - checks broker-reported binary/turbo availability without the known Digital
#   status path;
# - prefers regular markets and only falls back to OTC when no regular
#   candidates are broker-reported open;
# - records many quiet "shadow" predictions for learning;
# - resolves them after the 1-minute horizon;
# - trains a lightweight classifier from those real outcomes;
# - sends at most 4 signals per day.

import os, time, json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from iqoptionapi.stable_api import IQ_Option
except Exception as e:
    raise SystemExit(f"iqoptionapi import failed: {e}")

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise SystemExit(f"scikit-learn import failed: {e}")

EMAIL = os.environ.get("IQ_EMAIL", "")
PASSWORD = os.environ.get("IQ_PASSWORD", "")
TOKEN = os.environ.get("TELEGRAM_TOKEN") or os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("CHAT_ID", "")

MODE = "PRACTICE"
MAX_DAILY_SIGNALS = 4
MIN_CONFIDENCE = 0.78
LEARNING_MIN_EXAMPLES = 40
EXPIRY_SECONDS = 60
FRESH_SECONDS = 150

REGULAR_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
    "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "EURCAD", "AUDJPY"
]
OTC_PAIRS = [
    "EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "USDCHF-OTC",
    "AUDUSD-OTC", "USDCAD-OTC", "NZDUSD-OTC", "EURJPY-OTC",
    "GBPJPY-OTC", "EURGBP-OTC", "EURCAD-OTC", "AUDJPY-OTC"
]

STATE_FILE = Path("vetrax_learning_state.json")

FEATURES = [
    "ret1", "ret2", "ret3", "ret5", "body", "range", "upper_wick",
    "lower_wick", "ema9_gap", "ema21_gap", "ema50_gap", "rsi",
    "macd", "macd_signal", "atr_pct", "trend5", "momentum5",
    "break_high", "break_low", "vol_ratio", "hour_sin", "hour_cos"
]


def load_state():
    default = {"day": "", "daily_sent": 0, "examples": [], "pending": [], "sent": []}
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        for k, v in default.items():
            state.setdefault(k, v)
        return state
    except Exception:
        return default


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8"
    )


def telegram(text):
    if not TOKEN or not CHAT_ID:
        print(text)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=15
        )
    except Exception as e:
        print("Telegram error:", e)


def connect():
    if not EMAIL or not PASSWORD:
        raise SystemExit("Missing IQ_EMAIL or IQ_PASSWORD secrets.")
    iq = IQ_Option(EMAIL, PASSWORD)
    ok, reason = iq.connect()
    if not ok:
        raise SystemExit(f"IQ Option connection failed: {reason}")
    try:
        iq.change_balance(MODE)
    except Exception:
        pass
    print("Connected to IQ Option in PRACTICE mode.")
    return iq


def candles(iq, pair, n=120):
    try:
        raw = iq.get_candles(pair, 60, n, time.time())
        if not raw:
            return None
        df = pd.DataFrame(raw)
        if "from" in df:
            df["ts"] = pd.to_numeric(df["from"], errors="coerce")
        elif "at" in df:
            df["ts"] = pd.to_numeric(df["at"], errors="coerce")
        else:
            return None
        for col in ("open", "close", "min", "max"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" not in df:
            df["volume"] = 0
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        return (
            df.dropna(subset=["ts", "open", "close", "min", "max"])
              .sort_values("ts")
              .reset_index(drop=True)
        )
    except Exception as e:
        print(pair, "candle error:", e)
        return None


def load_binary_status(iq):
    """Read binary/turbo availability directly from initialization data.
    This deliberately does not use the Digital-market status endpoint.
    """
    try:
        raw = iq.get_all_init()
        if not isinstance(raw, dict):
            print("Broker initialization data unavailable.")
            return {}
        result = raw.get("result")
        if not isinstance(result, dict):
            print("Broker initialization result unavailable.")
            return {}

        status = {}
        for option in ("turbo", "binary"):
            block = result.get(option, {})
            actives = block.get("actives", {}) if isinstance(block, dict) else {}
            if not isinstance(actives, dict):
                continue

            for active_id, info in actives.items():
                if not isinstance(info, dict):
                    continue
                name = str(info.get("name", ""))
                if "." in name:
                    name = name.split(".", 1)[1]
                if not name:
                    continue

                enabled = info.get("enabled")
                suspended = info.get("is_suspended")

                # Accept boolean/numeric API variants.
                is_enabled = enabled is True or enabled == 1 or enabled == "1"
                is_suspended = suspended is True or suspended == 1 or suspended == "1"

                status.setdefault(name, {})[option] = {
                    "enabled": is_enabled,
                    "suspended": is_suspended,
                }

        open_regular = []
        open_otc = []
        for name, modes in status.items():
            open_here = any(
                isinstance(v, dict)
                and v.get("enabled") is True
                and v.get("suspended") is not True
                for v in modes.values()
            )
            if open_here:
                if name.endswith("-OTC"):
                    open_otc.append(name)
                else:
                    open_regular.append(name)

        print(
            f"Broker binary/turbo status: {len(open_regular)} regular open, "
            f"{len(open_otc)} OTC open."
        )
        if open_regular:
            print("Regular open sample:", ", ".join(sorted(open_regular)[:12]))
        if open_otc:
            print("OTC open sample:", ", ".join(sorted(open_otc)[:12]))

        return status
    except Exception as e:
        print("Broker availability check failed:", e)
        return {}


def broker_binary_open(detail, pair):
    item = detail.get(pair)
    if not isinstance(item, dict):
        return False
    return any(
        isinstance(info, dict)
        and info.get("enabled") is True
        and info.get("suspended") is not True
        for info in item.values()
    )


def fresh(df):
    if df is None or len(df) < 2:
        return False
    age = time.time() - float(df["ts"].iloc[-1])
    return 0 <= age <= FRESH_SECONDS


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def feature_row(df):
    if df is None or len(df) < 60:
        return None
    c, o, hi, lo = df["close"], df["open"], df["max"], df["min"]
    rng = (hi - lo).replace(0, np.nan)
    body = (c - o) / rng
    atr = (hi - lo).rolling(14).mean()
    e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
    macd, macd_signal = e9 - e21, ema(e9 - e21, 9)
    rr = rsi(c)
    vm = df["volume"].rolling(20).mean().replace(0, np.nan)

    vals = {
        "ret1": c.pct_change(1).iloc[-1],
        "ret2": c.pct_change(2).iloc[-1],
        "ret3": c.pct_change(3).iloc[-1],
        "ret5": c.pct_change(5).iloc[-1],
        "body": body.iloc[-1],
        "range": (rng / c).iloc[-1],
        "upper_wick": ((hi - np.maximum(o, c)) / rng).iloc[-1],
        "lower_wick": ((np.minimum(o, c) - lo) / rng).iloc[-1],
        "ema9_gap": c.iloc[-1] / e9.iloc[-1] - 1,
        "ema21_gap": c.iloc[-1] / e21.iloc[-1] - 1,
        "ema50_gap": c.iloc[-1] / e50.iloc[-1] - 1,
        "rsi": (rr.iloc[-1] - 50) / 50,
        "macd": macd.iloc[-1] / c.iloc[-1],
        "macd_signal": macd_signal.iloc[-1] / c.iloc[-1],
        "atr_pct": atr.iloc[-1] / c.iloc[-1],
        "trend5": (e9.iloc[-1] - e21.iloc[-1]) / c.iloc[-1],
        "momentum5": c.iloc[-1] / c.iloc[-6] - 1,
        "break_high": c.iloc[-1] / hi.iloc[-21:-1].max() - 1,
        "break_low": c.iloc[-1] / lo.iloc[-21:-1].min() - 1,
        "vol_ratio": df["volume"].iloc[-1] / vm.iloc[-1] if pd.notna(vm.iloc[-1]) else 1.0
    }
    dt = datetime.fromtimestamp(float(df["ts"].iloc[-1]), timezone.utc)
    vals["hour_sin"] = math.sin(2 * math.pi * dt.hour / 24)
    vals["hour_cos"] = math.cos(2 * math.pi * dt.hour / 24)
    arr = np.array([vals[k] for k in FEATURES], dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def rule_setup(df):
    if df is None or len(df) < 60:
        return None
    c, o, hi, lo = df["close"], df["open"], df["max"], df["min"]
    e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
    rr = float(rsi(c).iloc[-1])
    buy = sell = 0

    if e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1]:
        buy += 25
    elif e9.iloc[-1] < e21.iloc[-1] < e50.iloc[-1]:
        sell += 25
    if c.iloc[-1] > e9.iloc[-1]:
        buy += 10
    elif c.iloc[-1] < e9.iloc[-1]:
        sell += 10
    if 52 <= rr <= 70:
        buy += 12
    elif 30 <= rr <= 48:
        sell += 12
    if c.iloc[-1] > o.iloc[-1]:
        buy += 12
    elif c.iloc[-1] < o.iloc[-1]:
        sell += 12
    if c.iloc[-1] > c.iloc[-2]:
        buy += 10
    elif c.iloc[-1] < c.iloc[-2]:
        sell += 10

    rh, rl = hi.iloc[-21:-1].max(), lo.iloc[-21:-1].min()
    if c.iloc[-1] > rh:
        buy += 8
    elif c.iloc[-1] < rl:
        sell += 8

    if buy == sell:
        return None
    direction = "BUY" if buy > sell else "SELL"
    confidence = min(0.97, max(0.50, max(buy, sell) / 87.0))
    return direction, confidence


def train_predict(state, x):
    examples = state["examples"]
    if len(examples) < LEARNING_MIN_EXAMPLES:
        return None
    X = np.asarray([e["x"] for e in examples], dtype=float)
    y = np.asarray([e["y"] for e in examples], dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    try:
        scaler = StandardScaler()
        xs = scaler.fit_transform(X)
        model = SGDClassifier(
            loss="log_loss", alpha=0.001, max_iter=2000,
            tol=1e-4, random_state=42, class_weight="balanced"
        )
        model.fit(xs, y)
        return float(model.predict_proba(scaler.transform([x]))[0, 1])
    except Exception as e:
        print("ML training skipped:", e)
        return None


def resolve_pending(state, iq):
    if not state["pending"]:
        return
    remaining = []
    changed = False

    for item in state["pending"]:
        if time.time() < item["resolve_at"]:
            remaining.append(item)
            continue

        df = candles(iq, item["pair"], 15)
        if df is None:
            remaining.append(item)
            continue

        future = df[df["ts"] >= item["entry_ts"] + EXPIRY_SECONDS]
        if future.empty:
            remaining.append(item)
            continue

        exit_price = float(future.iloc[0]["close"])
        entry_price = float(item["entry_price"])

        if exit_price == entry_price:
            y = 0
        elif item["direction"] == "BUY":
            y = int(exit_price > entry_price)
        else:
            y = int(exit_price < entry_price)

        state["examples"].append({
            "x": item["x"],
            "y": y,
            "pair": item["pair"],
            "direction": item["direction"],
            "time": item["time"]
        })

        if item.get("sent"):
            telegram(
                "🧠 VETRA-X LEARNING RESULT\n\n"
                f"{item['pair']} {item['direction']}\n"
                f"RESULT: {'WIN' if y else 'LOSS'}\n"
                f"Training examples: {len(state['examples'])}"
            )

        changed = True

    state["pending"] = remaining[-1000:]
    state["examples"] = state["examples"][-3000:]
    if changed:
        save_state(state)


def add_pending(state, item, sent=False):
    state["pending"].append({
        "pair": item["pair"],
        "direction": item["direction"],
        "x": item["x"].tolist(),
        "entry_price": float(item["entry_price"]),
        "entry_ts": float(item["entry_ts"]),
        "resolve_at": time.time() + 70,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sent": bool(sent)
    })


def recently_sent(state, pair):
    cutoff = time.time() - 600
    return any(
        s.get("pair") == pair and s.get("created_at", 0) >= cutoff
        for s in state["sent"]
    )


def main():
    state = load_state()
    today = datetime.now().strftime("%Y-%m-%d")

    if state["day"] != today:
        state["day"] = today
        state["daily_sent"] = 0
        state["sent"] = []
        save_state(state)

    iq = connect()
    resolve_pending(state, iq)
    broker_detail = load_binary_status(iq)

    # Use broker-reported assets dynamically. This avoids hardcoded OTC
    # names that the installed API may not know.
    regular_open = sorted(
        p for p in broker_detail
        if not p.endswith("-OTC") and broker_binary_open(broker_detail, p)
    )
    otc_open = sorted(
        p for p in broker_detail
        if p.endswith("-OTC") and broker_binary_open(broker_detail, p)
    )

    # Prefer the familiar major/JPY pairs, then allow any broker-reported pair.
    preferred_regular = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
        "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "EURCAD", "AUDJPY"
    ]
    preferred_otc = [
        "EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "USDCHF-OTC",
        "NZDUSD-OTC", "EURJPY-OTC", "GBPJPY-OTC", "EURGBP-OTC",
        "AUDJPY-OTC", "GBPCHF-OTC", "NZDJPY-OTC", "NZDCAD-OTC"
    ]

    regular_order = [p for p in preferred_regular if p in regular_open]
    regular_order += [p for p in regular_open if p not in regular_order]

    otc_order = [p for p in preferred_otc if p in otc_open]
    otc_order += [p for p in otc_open if p not in otc_order]

    candidates = []
    for pair in regular_order:
        df = candles(iq, pair)
        if fresh(df):
            candidates.append(("REGULAR", pair, df))

    # OTC is fallback only when no regular market has fresh candles.
    if not candidates:
        for pair in otc_order:
            df = candles(iq, pair)
            if fresh(df):
                candidates.append(("OTC", pair, df))

    if not candidates:
        print("No broker-reported open pair with fresh candles. No signal.")
        return

    ranked = []

    for market, pair, df in candidates:
        setup = rule_setup(df)
        x = feature_row(df)
        if setup is None or x is None:
            continue

        direction, rule_conf = setup
        ml = train_predict(state, x)

        if ml is None:
            final_direction = direction
            final_conf = rule_conf
            model_text = "learning warm-up"
        else:
            ml_direction = "BUY" if ml >= 0.5 else "SELL"
            ml_conf = max(ml, 1 - ml)
            if ml_direction != direction:
                continue
            final_direction = ml_direction
            final_conf = 0.55 * rule_conf + 0.45 * ml_conf
            model_text = f"ML {ml_conf * 100:.1f}%"

        ranked.append({
            "market": market,
            "pair": pair,
            "x": x,
            "direction": final_direction,
            "confidence": float(final_conf),
            "model_text": model_text,
            "entry_ts": float(df["ts"].iloc[-1]),
            "entry_price": float(df["close"].iloc[-1])
        })

    if not ranked:
        print("No setup where the learning model and rule engine agree.")
        return

    ranked.sort(key=lambda z: z["confidence"], reverse=True)

    # Record all strong candidates for learning, even when they are not sent.
    for item in ranked:
        if item["confidence"] >= 0.55:
            add_pending(state, item, sent=False)

    best = ranked[0]

    if state["daily_sent"] >= MAX_DAILY_SIGNALS:
        save_state(state)
        print("Daily signal limit reached.")
        return

    if best["confidence"] < MIN_CONFIDENCE:
        save_state(state)
        print(f"Best setup below threshold: {best['confidence'] * 100:.1f}%")
        return

    if recently_sent(state, best["pair"]):
        save_state(state)
        print("Best pair was recently signalled. No duplicate signal.")
        return

    # Mark the matching shadow observation as the visible signal.
    for item in reversed(state["pending"]):
        if (
            item["pair"] == best["pair"]
            and item["entry_ts"] == best["entry_ts"]
            and item["sent"] is False
        ):
            item["sent"] = True
            break

    state["daily_sent"] += 1
    state["sent"].append({
        "pair": best["pair"],
        "direction": best["direction"],
        "confidence": best["confidence"],
        "created_at": time.time()
    })
    state["sent"] = state["sent"][-100:]
    save_state(state)

    entry_time = datetime.fromtimestamp(
        best["entry_ts"], timezone.utc
    ).strftime("%I:%M %p")

    label = " (OTC)" if best["market"] == "OTC" else " (REGULAR)"
    message = (
        "🧠 VETRA-X LEARNING SIGNAL\n\n"
        f"PAIR: {best['pair']}{label}\n"
        "EXPIRY: 1 MIN\n"
        f"CONFIDENCE: {best['confidence'] * 100:.1f}%\n"
        f"ENTRY: {entry_time}\n"
        f"DIRECTION: {'🟢 BUY' if best['direction'] == 'BUY' else '🔴 SELL'}\n\n"
        f"MODEL: {best['model_text']}\n"
        f"TRAINING EXAMPLES: {len(state['examples'])}\n"
        f"TODAY: {state['daily_sent']}/{MAX_DAILY_SIGNALS}\n"
        "PRACTICE / SIGNAL ONLY"
    )
    telegram(message)
    print(message)


if __name__ == "__main__":
    main()
