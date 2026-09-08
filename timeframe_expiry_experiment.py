"""
V1 TIMEFRAME + EXPIRY EXPERIMENT

Research only. Uses the V1 scanner's indicator/entry rules, while changing
ONLY the entry timeframe and fixed expiry so we can identify which
combination is worth optimizing next.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import iqoptionapi.constants as OP_code
from iqoptionapi.stable_api import IQ_Option
import scanner as strategy

PAIRS = strategy.REGULAR_PAIRS
ENTRY_TIMEFRAME = os.getenv("ENTRY_TIMEFRAME", "1M").upper()
EXPIRY_MINUTES = int(os.getenv("EXPIRY_MINUTES", "10"))
HISTORY_MINUTES = int(os.getenv("BACKTEST_MINUTES", "14400"))
END_OFFSET_MINUTES = int(os.getenv("BACKTEST_END_OFFSET_MINUTES", "10080"))
MAX_SIGNALS_PER_PAIR = int(os.getenv("BACKTEST_MAX_SIGNALS_PER_PAIR", "150"))
MIN_GAP_MINUTES = int(os.getenv("BACKTEST_SIGNAL_GAP", "3"))
CANDLES_PER_REQUEST = 1000

TIMEFRAMES = {"1M": 60, "5M": 300, "15M": 900}

@dataclass
class Result:
    pair: str
    direction: str
    entry: float
    exit_price: float
    score: int
    confidence: float
    win: bool


def server_time(iq: IQ_Option) -> int:
    try:
        return int(float(iq.timesync.server_timestamp))
    except Exception:
        return int(time.time())


def map_option_symbols(iq: IQ_Option) -> Dict[str, int]:
    found: Dict[str, int] = {}
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
            raw = str(active.get("name", "")).strip()
            if not raw:
                continue
            name = raw.split(".")[-1].upper()
            try:
                aid = int(active_id)
                OP_code.ACTIVES[name] = aid
                found[name] = aid
            except Exception:
                pass
    return found


def discover_otc(iq: IQ_Option, pair: str) -> Optional[str]:
    target = pair.upper() + "-OTC"
    data = iq.get_all_init_v2()
    if not isinstance(data, dict):
        return None
    for option_type in ("binary", "turbo"):
        section = data.get(option_type, {})
        actives = section.get("actives", {}) if isinstance(section, dict) else {}
        if not isinstance(actives, dict):
            continue
        for active_id, active in actives.items():
            if not isinstance(active, dict):
                continue
            name = str(active.get("name", "")).strip().split(".")[-1].upper()
            try:
                OP_code.ACTIVES[name] = int(active_id)
            except Exception:
                pass
            if name == target:
                return name
    return None


def raw_to_candles(raw: Sequence[Dict[str, Any]]) -> List[strategy.Candle]:
    out: List[strategy.Candle] = []
    for item in raw or []:
        try:
            ts = int(float(item.get("to", item.get("from"))))
            if ts > 10_000_000_000:
                ts //= 1000
            out.append(strategy.Candle(ts, float(item["open"]), float(item["close"]),
                                       float(item["max"]), float(item["min"]),
                                       float(item.get("volume", 0.0))))
        except Exception:
            continue
    out.sort(key=lambda c: c.timestamp)
    unique = {c.timestamp: c for c in out}
    return [unique[k] for k in sorted(unique)]


def fetch_history(iq: IQ_Option, symbol: str, seconds: int, minutes_needed: int) -> List[strategy.Candle]:
    target = max(strategy.MIN_CANDLES + 20, int(minutes_needed * 60 / seconds) + 150)
    rows: Dict[int, strategy.Candle] = {}
    end = server_time(iq) - END_OFFSET_MINUTES * 60
    attempts = 0
    while len(rows) < target and attempts < 15:
        attempts += 1
        try:
            raw = iq.get_candles(symbol, seconds, CANDLES_PER_REQUEST, end)
        except Exception:
            raw = []
        batch = raw_to_candles(raw)
        if not batch:
            break
        before = len(rows)
        for c in batch:
            rows[c.timestamp] = c
        if len(rows) == before:
            break
        end = batch[0].timestamp - seconds
        time.sleep(0.12)
    return [rows[k] for k in sorted(rows)][-target:]


def bias(candles: Sequence[strategy.Candle]) -> str:
    return strategy.bias_from_trend([c.close for c in candles])


def analyze_v1(c_entry: Sequence[strategy.Candle], confirmations: Sequence[Sequence[strategy.Candle]]) -> Optional[Tuple[str, int, float]]:
    if len(c_entry) < strategy.MIN_CANDLES or any(len(c) < strategy.MIN_CANDLES for c in confirmations):
        return None
    closes = [c.close for c in c_entry]
    entry_bias, _ = strategy.entry_structure(c_entry)
    if entry_bias == "NEUTRAL":
        return None
    direction = "CALL" if entry_bias == "BULLISH" else "PUT"
    if any(bias(c) != entry_bias for c in confirmations):
        return None

    rsi_value = strategy.rsi(closes, 14)
    atr_value = strategy.atr(c_entry, 14)
    if rsi_value is None or atr_value is None or not strategy.volatility_ok(c_entry, atr_value):
        return None
    if direction == "CALL" and not (50 <= rsi_value <= 72):
        return None
    if direction == "PUT" and not (28 <= rsi_value <= 50):
        return None

    score = 36
    confirmations_count = 2
    conflicts = 0
    if direction == "CALL":
        if 54 <= rsi_value <= 67: score += 10; confirmations_count += 1
        elif rsi_value < 54: score += 5
        else: conflicts += 1
    else:
        if 33 <= rsi_value <= 46: score += 10; confirmations_count += 1
        elif rsi_value > 46: score += 5
        else: conflicts += 1

    if strategy.macd_confirmation(closes, direction): score += 12; confirmations_count += 1
    else: conflicts += 1
    score += 8; confirmations_count += 1

    mom = strategy.candle_momentum(c_entry)
    if (mom >= 0.22 if direction == "CALL" else mom <= -0.22): score += 10; confirmations_count += 1
    else: conflicts += 1

    if strategy.two_candle_confirmation(c_entry, direction): score += 14; confirmations_count += 1
    else: conflicts += 1

    sr_ok, _ = strategy.near_support_resistance(c_entry, direction, atr_value)
    score += 10 if sr_ok else 2
    if sr_ok: confirmations_count += 1

    # Preserve V1's higher-timeframe momentum confirmation.
    for higher in confirmations:
        hm = strategy.candle_momentum(higher)
        if direction == "CALL":
            if hm > -0.15: score += 3; confirmations_count += 1
            else: conflicts += 1
        else:
            if hm < 0.15: score += 3; confirmations_count += 1
            else: conflicts += 1

    score = max(0, min(100, int(round(score))))
    confidence = strategy.confidence_from_score(score, confirmations_count, conflicts)
    if score < strategy.MIN_SCORE or confidence < strategy.MIN_CONFIDENCE:
        return None
    return direction, EXPIRY_MINUTES, confidence


def candle_at_or_before(candles: Sequence[strategy.Candle], ts: int) -> Optional[strategy.Candle]:
    lo, hi = 0, len(candles) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if candles[mid].timestamp <= ts:
            best = candles[mid]; lo = mid + 1
        else: hi = mid - 1
    return best


def run_pair(iq: IQ_Option, pair: str, symbol: str) -> List[Result]:
    entry_seconds = TIMEFRAMES[ENTRY_TIMEFRAME]
    higher = {"1M": ["5M", "15M"], "5M": ["15M"], "15M": []}[ENTRY_TIMEFRAME]
    needed = HISTORY_MINUTES + EXPIRY_MINUTES + 30
    entry = fetch_history(iq, symbol, entry_seconds, needed)
    higher_data = {tf: fetch_history(iq, symbol, TIMEFRAMES[tf], needed) for tf in higher}
    if len(entry) < strategy.MIN_CANDLES or any(len(v) < strategy.MIN_CANDLES for v in higher_data.values()):
        return []

    results: List[Result] = []
    last_signal = -10**9
    for i in range(strategy.MIN_CANDLES - 1, len(entry)):
        cur = entry[i]
        if cur.timestamp - last_signal < MIN_GAP_MINUTES * 60:
            continue
        exit_ts = cur.timestamp + EXPIRY_MINUTES * 60
        if exit_ts > entry[-1].timestamp:
            continue
        conf = []
        for tf in higher:
            conf.append([c for c in higher_data[tf] if c.timestamp <= cur.timestamp])
        signal = analyze_v1(entry[:i+1], conf)
        if signal is None:
            continue
        direction, _, confidence = signal
        exit_candle = candle_at_or_before(entry, exit_ts)
        if exit_candle is None or exit_candle.timestamp < exit_ts:
            continue
        win = exit_candle.close > cur.close if direction == "CALL" else exit_candle.close < cur.close
        score = int(round(max(0, min(100, confidence * 100))))
        results.append(Result(pair, direction, cur.close, exit_candle.close, score, confidence, win))
        last_signal = cur.timestamp
        if len(results) >= MAX_SIGNALS_PER_PAIR:
            break
    return results


def main() -> None:
    email, password = os.getenv("IQ_EMAIL", "").strip(), os.getenv("IQ_PASSWORD", "").strip()
    if not email or not password:
        raise RuntimeError("IQ_EMAIL and IQ_PASSWORD are required")
    if ENTRY_TIMEFRAME not in TIMEFRAMES:
        raise RuntimeError(f"Unsupported ENTRY_TIMEFRAME={ENTRY_TIMEFRAME}")
    iq = IQ_Option(email, password)
    ok, reason = iq.connect()
    if not ok: raise RuntimeError(f"IQ Option connection failed: {reason}")
    iq.change_balance("PRACTICE")
    print(f"Connected to IQ Option PRACTICE. Entry={ENTRY_TIMEFRAME}, Expiry={EXPIRY_MINUTES}m")
    print(f"Validation window ends: {END_OFFSET_MINUTES} minutes before current server time")
    map_option_symbols(iq)
    all_results: List[Result] = []
    for pair in PAIRS:
        symbol = pair
        try:
            test = fetch_history(iq, symbol, TIMEFRAMES[ENTRY_TIMEFRAME], 180)
        except Exception:
            test = []
        if len(test) < strategy.MIN_CANDLES:
            otc = discover_otc(iq, pair)
            if otc: symbol = otc
            else: print(f"{pair}: no usable symbol"); continue
        print(f"Testing {pair} -> {symbol}...")
        try:
            rows = run_pair(iq, pair, symbol)
            all_results.extend(rows)
            w = sum(r.win for r in rows)
            print(f"  Result: {w}/{len(rows)} wins ({(w/len(rows)*100 if rows else 0):.2f}%)")
        except Exception as exc:
            print(f"  ERROR: {exc}")
    print("\n" + "="*64)
    print("V1 TIMEFRAME + EXPIRY EXPERIMENT")
    print("="*64)
    print(f"Entry timeframe: {ENTRY_TIMEFRAME}")
    print(f"Expiry:           {EXPIRY_MINUTES} min")
    print(f"Signals:          {len(all_results)}")
    if all_results:
        wins = sum(r.win for r in all_results)
        print(f"Wins:             {wins}")
        print(f"Losses:           {len(all_results)-wins}")
        print(f"Accuracy:         {wins/len(all_results)*100:.2f}%")
        for d in ("CALL", "PUT"):
            s = [r for r in all_results if r.direction == d]
            if s: print(f"{d}:               {sum(r.win for r in s)}/{len(s)} = {sum(r.win for r in s)/len(s)*100:.2f}%")
    print("\nThis experiment changes timeframe/expiry only; it is not a claim of future performance.")

if __name__ == "__main__":
    main()
