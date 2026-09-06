"""
IQ Option Strategy Backtester

Uses the SAME indicator/entry/scoring rules as scanner.py, but tests them
against historical candles instead of live candles.

This is research only. It does NOT place trades and does NOT send Telegram
signals. Results are historical and are not a guarantee of future accuracy.
"""
from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import iqoptionapi.constants as OP_code
from iqoptionapi.stable_api import IQ_Option

# Import the actual live scanner so the backtest uses its real strategy rules.
import scanner as strategy


PAIRS = strategy.REGULAR_PAIRS
TIMEFRAMES = strategy.TIMEFRAMES
CANDLES_PER_REQUEST = 1000
HISTORY_MINUTES = int(os.getenv("BACKTEST_MINUTES", "2880"))  # 48 hours
MAX_SIGNALS_PER_PAIR = int(os.getenv("BACKTEST_MAX_SIGNALS_PER_PAIR", "40"))
MIN_GAP_MINUTES = int(os.getenv("BACKTEST_SIGNAL_GAP", "3"))


@dataclass
class Result:
    pair: str
    market: str
    direction: str
    expiry: int
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
            raw = str(active.get("name", "")).strip()
            name = raw.split(".")[-1].upper()
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
            out.append(strategy.Candle(
                timestamp=ts,
                open=float(item["open"]),
                close=float(item["close"]),
                high=float(item["max"]),
                low=float(item["min"]),
                volume=float(item.get("volume", 0.0)),
            ))
        except Exception:
            continue
    out.sort(key=lambda c: c.timestamp)
    # Remove duplicate timestamps.
    unique: Dict[int, strategy.Candle] = {}
    for c in out:
        unique[c.timestamp] = c
    return [unique[k] for k in sorted(unique)]


def fetch_history(iq: IQ_Option, symbol: str, seconds: int, minutes_needed: int) -> List[strategy.Candle]:
    target = max(150, int(minutes_needed * 60 / seconds) + 150)
    all_rows: Dict[int, strategy.Candle] = {}
    end = server_time(iq)
    attempts = 0

    while len(all_rows) < target and attempts < 12:
        attempts += 1
        try:
            raw = iq.get_candles(symbol, seconds, CANDLES_PER_REQUEST, end)
        except Exception:
            raw = []
        batch = raw_to_candles(raw)
        if not batch:
            break
        before = len(all_rows)
        for c in batch:
            all_rows[c.timestamp] = c
        if len(all_rows) == before:
            break
        oldest = batch[0].timestamp
        end = oldest - seconds
        time.sleep(0.15)

    candles = [all_rows[k] for k in sorted(all_rows)]
    return candles[-target:]


def trend_bias_at(candles: Sequence[strategy.Candle]) -> str:
    closes = [c.close for c in candles]
    return strategy.bias_from_trend(closes)


def analyze_historical(
    c1: Sequence[strategy.Candle],
    c5: Sequence[strategy.Candle],
    c15: Sequence[strategy.Candle],
) -> Optional[Tuple[str, int, float]]:
    """Same strategy gates/scoring as scanner.py, without live freshness checks."""
    if len(c1) < strategy.MIN_CANDLES or len(c5) < strategy.MIN_CANDLES or len(c15) < strategy.MIN_CANDLES:
        return None

    closes1 = [c.close for c in c1]
    closes5 = [c.close for c in c5]
    closes15 = [c.close for c in c15]

    b5 = trend_bias_at(c5)
    b15 = trend_bias_at(c15)
    entry_bias, _ = strategy.entry_structure(c1)
    if entry_bias == "NEUTRAL":
        return None
    direction = "CALL" if entry_bias == "BULLISH" else "PUT"
    if direction == "CALL" and not (b5 == "BULLISH" and b15 == "BULLISH"):
        return None
    if direction == "PUT" and not (b5 == "BEARISH" and b15 == "BEARISH"):
        return None

    rsi_value = strategy.rsi(closes1, 14)
    atr_value = strategy.atr(c1, 14)
    if rsi_value is None or atr_value is None or not strategy.volatility_ok(c1, atr_value):
        return None
    if direction == "CALL" and not (50.0 <= rsi_value <= 72.0):
        return None
    if direction == "PUT" and not (28.0 <= rsi_value <= 50.0):
        return None

    score = 36
    confirmations = 2
    conflicts = 0

    if direction == "CALL":
        if 54 <= rsi_value <= 67:
            score += 10; confirmations += 1
        elif 50 <= rsi_value < 54:
            score += 5
        else:
            conflicts += 1
    else:
        if 33 <= rsi_value <= 46:
            score += 10; confirmations += 1
        elif 46 < rsi_value <= 50:
            score += 5
        else:
            conflicts += 1

    if strategy.macd_confirmation(closes1, direction):
        score += 12; confirmations += 1
    else:
        conflicts += 1

    score += 8; confirmations += 1

    mom = strategy.candle_momentum(c1)
    if (mom >= 0.22 if direction == "CALL" else mom <= -0.22):
        score += 10; confirmations += 1
    else:
        conflicts += 1

    if strategy.two_candle_confirmation(c1, direction):
        score += 14; confirmations += 1
    else:
        conflicts += 1

    sr_ok, _ = strategy.near_support_resistance(c1, direction, atr_value)
    if sr_ok:
        score += 10; confirmations += 1
    else:
        score += 2

    m5 = strategy.candle_momentum(c5)
    m15 = strategy.candle_momentum(c15)
    if direction == "CALL":
        if m5 > -0.15 and m15 > -0.15:
            score += 5; confirmations += 1
        else:
            conflicts += 1
    else:
        if m5 < 0.15 and m15 < 0.15:
            score += 5; confirmations += 1
        else:
            conflicts += 1

    score = max(0, min(100, int(round(score))))
    confidence = strategy.confidence_from_score(score, confirmations, conflicts)
    if score < strategy.MIN_SCORE or confidence < strategy.MIN_CONFIDENCE:
        return None

    expiry = strategy.choose_expiry(c1, atr_value, score)
    return direction, expiry, confidence


def candle_at_or_before(candles: Sequence[strategy.Candle], ts: int) -> Optional[strategy.Candle]:
    # Binary search would be faster, but these lists are small enough and this
    # keeps the script simple/reliable.
    lo, hi = 0, len(candles) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if candles[mid].timestamp <= ts:
            best = candles[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def run_pair(iq: IQ_Option, pair: str, symbol: str, market: str) -> List[Result]:
    c1 = fetch_history(iq, symbol, 60, HISTORY_MINUTES)
    c5 = fetch_history(iq, symbol, 300, HISTORY_MINUTES)
    c15 = fetch_history(iq, symbol, 900, HISTORY_MINUTES)
    if len(c1) < strategy.MIN_CANDLES or len(c5) < strategy.MIN_CANDLES or len(c15) < strategy.MIN_CANDLES:
        print(f"{pair} {market}: insufficient history ({len(c1)}/{len(c5)}/{len(c15)})")
        return []

    results: List[Result] = []
    last_signal_ts = -10**9
    # Only use 1M candles where the full future expiry exists.
    for i in range(strategy.MIN_CANDLES - 1, len(c1) - 12):
        cur = c1[i]
        if cur.timestamp - last_signal_ts < MIN_GAP_MINUTES * 60:
            continue

        # Historical 5M/15M context ending at or before this 1M candle.
        h5 = [c for c in c5 if c.timestamp <= cur.timestamp]
        h15 = [c for c in c15 if c.timestamp <= cur.timestamp]
        if len(h5) < strategy.MIN_CANDLES or len(h15) < strategy.MIN_CANDLES:
            continue

        signal = analyze_historical(c1[:i+1], h5, h15)
        if signal is None:
            continue
        direction, expiry, confidence = signal

        exit_index = i + expiry
        if exit_index >= len(c1):
            continue
        exit_price = c1[exit_index].close
        win = exit_price > cur.close if direction == "CALL" else exit_price < cur.close

        # Recreate score for reporting by approximating it from the scanner
        # signal logic: use the same function inputs and infer score from
        # confidence is not exact. So calculate the exact score independently
        # through a lightweight replay by matching scanner's formula.
        # The report's primary metric is W/L, not the displayed confidence.
        score = score_for_report(c1[:i+1], h5, h15, direction)

        results.append(Result(pair, market, direction, expiry, cur.close, exit_price, score, confidence, win))
        last_signal_ts = cur.timestamp
        if len(results) >= MAX_SIGNALS_PER_PAIR:
            break

    return results


def score_for_report(c1: Sequence[strategy.Candle], c5: Sequence[strategy.Candle], c15: Sequence[strategy.Candle], direction: str) -> int:
    closes1 = [c.close for c in c1]
    r = strategy.rsi(closes1, 14)
    a = strategy.atr(c1, 14)
    if r is None or a is None:
        return 0
    score = 36
    if direction == "CALL":
        score += 10 if 54 <= r <= 67 else (5 if 50 <= r < 54 else 0)
    else:
        score += 10 if 33 <= r <= 46 else (5 if 46 < r <= 50 else 0)
    score += 12 if strategy.macd_confirmation(closes1, direction) else 0
    score += 8
    m = strategy.candle_momentum(c1)
    score += 10 if (m >= .22 if direction == "CALL" else m <= -.22) else 0
    score += 14 if strategy.two_candle_confirmation(c1, direction) else 0
    ok, _ = strategy.near_support_resistance(c1, direction, a)
    score += 10 if ok else 2
    m5, m15 = strategy.candle_momentum(c5), strategy.candle_momentum(c15)
    score += 5 if ((m5 > -.15 and m15 > -.15) if direction == "CALL" else (m5 < .15 and m15 < .15)) else 0
    return max(0, min(100, int(round(score))))


def print_report(results: List[Result]) -> None:
    print("\n" + "=" * 64)
    print("IQ OPTION STRATEGY BACKTEST REPORT")
    print("=" * 64)
    print(f"History requested: {HISTORY_MINUTES} minutes")
    print(f"Signals tested:     {len(results)}")
    if not results:
        print("No qualifying historical signals were found.")
        return

    wins = sum(r.win for r in results)
    losses = len(results) - wins
    accuracy = wins / len(results) * 100
    print(f"Wins:               {wins}")
    print(f"Losses:             {losses}")
    print(f"Historical accuracy:{accuracy:.2f}%")

    for expiry in sorted(set(r.expiry for r in results)):
        subset = [r for r in results if r.expiry == expiry]
        w = sum(r.win for r in subset)
        print(f"  {expiry:>2} min expiry: {w}/{len(subset)} = {w/len(subset)*100:.2f}%")

    for direction in ("CALL", "PUT"):
        subset = [r for r in results if r.direction == direction]
        if subset:
            w = sum(r.win for r in subset)
            print(f"  {direction}: {w}/{len(subset)} = {w/len(subset)*100:.2f}%")

    print("\nPair breakdown:")
    for pair in sorted(set(r.pair for r in results)):
        subset = [r for r in results if r.pair == pair]
        w = sum(r.win for r in subset)
        print(f"  {pair:7s} {w:>2}/{len(subset):<2} = {w/len(subset)*100:6.2f}%")

    print("\nIMPORTANT: the scanner's displayed confidence is NOT a measured win probability.")
    print("Do not use a real-money account based on this report alone.")


def main() -> None:
    email = os.getenv("IQ_EMAIL", "").strip()
    password = os.getenv("IQ_PASSWORD", "").strip()
    if not email or not password:
        raise RuntimeError("IQ_EMAIL and IQ_PASSWORD are required.")

    iq = IQ_Option(email, password)
    ok, reason = iq.connect()
    if not ok:
        raise RuntimeError(f"IQ Option connection failed: {reason}")
    iq.change_balance("PRACTICE")
    print("Connected to IQ Option PRACTICE. No trades will be placed.")

    try:
        mapping = map_option_symbols(iq)
        print(f"Loaded {len(mapping)} binary/turbo symbols.")
        all_results: List[Result] = []

        for pair in PAIRS:
            # Prefer regular if it has usable historical candles; otherwise OTC.
            symbol = pair
            market = "REGULAR"
            try:
                test = fetch_history(iq, symbol, 60, 180)
            except Exception:
                test = []
            if len(test) < strategy.MIN_CANDLES:
                otc = discover_otc(iq, pair)
                if otc:
                    symbol, market = otc, "OTC"
                else:
                    print(f"{pair}: no usable regular/OTC symbol")
                    continue

            print(f"\nTesting {pair} -> {symbol} ({market})...")
            try:
                results = run_pair(iq, pair, symbol, market)
                all_results.extend(results)
                if results:
                    w = sum(r.win for r in results)
                    print(f"  Result: {w}/{len(results)} wins ({w/len(results)*100:.2f}%)")
                else:
                    print("  Result: no qualifying signals")
            except Exception as exc:
                print(f"  ERROR: {exc}")

        print_report(all_results)
    finally:
        try:
            iq.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
