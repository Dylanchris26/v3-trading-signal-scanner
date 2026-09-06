"""
IQ Option Strategy V2 Backtester

Tests scanner_v2.py against historical 1M/5M/15M candles.
Research only: no trades, no Telegram messages.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import iqoptionapi.constants as OP_code
from iqoptionapi.stable_api import IQ_Option
import scanner_v2 as strategy

PAIRS = strategy.REGULAR_PAIRS
CANDLES_PER_REQUEST = 1000
HISTORY_MINUTES = int(os.getenv("BACKTEST_MINUTES", "10080"))  # 7 days
MAX_SIGNALS_PER_PAIR = int(os.getenv("BACKTEST_MAX_SIGNALS_PER_PAIR", "60"))
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
    quality: float
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
            except (TypeError, ValueError):
                continue
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
            if not raw:
                continue
            name = raw.split(".")[-1].upper()
            try:
                OP_code.ACTIVES[name] = int(active_id)
            except (TypeError, ValueError):
                pass
            if name == target:
                return name
    return None


def raw_to_candles(raw: Sequence[Dict[str, Any]]) -> List[strategy.Candle]:
    out: List[strategy.Candle] = []
    for item in raw or []:
        try:
            value = item.get("to", item.get("from"))
            if value is None:
                continue
            ts = int(float(value))
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
        except (KeyError, TypeError, ValueError):
            continue

    out.sort(key=lambda c: c.timestamp)
    unique: Dict[int, strategy.Candle] = {}
    for candle in out:
        unique[candle.timestamp] = candle
    return [unique[k] for k in sorted(unique)]


def fetch_history(
    iq: IQ_Option,
    symbol: str,
    seconds: int,
    minutes_needed: int,
) -> List[strategy.Candle]:
    target = max(
        150,
        int(minutes_needed * 60 / seconds) + 150,
    )
    rows: Dict[int, strategy.Candle] = {}
    end = server_time(iq)

    for _ in range(20):
        if len(rows) >= target:
            break
        try:
            raw = iq.get_candles(
                symbol,
                seconds,
                CANDLES_PER_REQUEST,
                end,
            )
        except Exception:
            raw = []

        batch = raw_to_candles(raw)
        if not batch:
            break

        before = len(rows)
        for candle in batch:
            rows[candle.timestamp] = candle
        if len(rows) == before:
            break

        end = batch[0].timestamp - seconds
        time.sleep(0.12)

    return [rows[k] for k in sorted(rows)][-target:]


def context_at_or_before(
    candles: Sequence[strategy.Candle],
    timestamp: int,
) -> List[strategy.Candle]:
    # The lists are small enough for this simple filtering approach.
    return [c for c in candles if c.timestamp <= timestamp]


def run_pair(
    iq: IQ_Option,
    pair: str,
    symbol: str,
    market: str,
) -> List[Result]:
    c1 = fetch_history(iq, symbol, 60, HISTORY_MINUTES)
    c5 = fetch_history(iq, symbol, 300, HISTORY_MINUTES)
    c15 = fetch_history(iq, symbol, 900, HISTORY_MINUTES)

    if min(len(c1), len(c5), len(c15)) < strategy.MIN_CANDLES:
        print(
            f"{pair} {market}: insufficient history "
            f"({len(c1)}/{len(c5)}/{len(c15)})"
        )
        return []

    results: List[Result] = []
    last_signal_ts = -10**12

    # Need enough future 1M candles for the fixed 10-minute expiry.
    final_index = len(c1) - strategy.V2_EXPIRY_MINUTES - 1

    for i in range(strategy.MIN_CANDLES - 1, final_index + 1):
        current = c1[i]
        if current.timestamp - last_signal_ts < MIN_GAP_MINUTES * 60:
            continue

        h5 = context_at_or_before(c5, current.timestamp)
        h15 = context_at_or_before(c15, current.timestamp)
        if len(h5) < strategy.MIN_CANDLES or len(h15) < strategy.MIN_CANDLES:
            continue

        setup = strategy.analyze_setup(
            c1[: i + 1],
            h5,
            h15,
        )
        if setup is None:
            continue

        expiry = int(setup["expiry"])
        exit_index = i + expiry
        if exit_index >= len(c1):
            continue

        entry = current.close
        exit_price = c1[exit_index].close
        direction = str(setup["direction"])
        win = (
            exit_price > entry
            if direction == "CALL"
            else exit_price < entry
        )

        results.append(Result(
            pair=pair,
            market=market,
            direction=direction,
            expiry=expiry,
            entry=entry,
            exit_price=exit_price,
            score=int(setup["score"]),
            quality=float(setup["confidence"]),
            win=win,
        ))
        last_signal_ts = current.timestamp

        if len(results) >= MAX_SIGNALS_PER_PAIR:
            break

    return results


def pct(wins: int, total: int) -> str:
    return "0.00%" if total == 0 else f"{wins / total * 100:.2f}%"


def print_report(results: List[Result]) -> None:
    print("\n" + "=" * 68)
    print("IQ OPTION V2 STRATEGY BACKTEST REPORT")
    print("=" * 68)
    print(f"History requested: {HISTORY_MINUTES} minutes")
    print(f"Pairs tested:      {len(PAIRS)}")
    print(f"Signals tested:    {len(results)}")

    if not results:
        print("No qualifying V2 historical signals were found.")
        print("The V2 gates may simply be too strict for this sample.")
        return

    wins = sum(r.win for r in results)
    losses = len(results) - wins
    accuracy = wins / len(results) * 100
    baseline = 54.69
    print(f"Wins:              {wins}")
    print(f"Losses:            {losses}")
    print(f"Historical accuracy: {accuracy:.2f}%")
    print(f"Baseline V1:         {baseline:.2f}%")
    print(f"Change vs V1:        {accuracy - baseline:+.2f} percentage points")

    print("\nExpiry breakdown:")
    for expiry in sorted(set(r.expiry for r in results)):
        subset = [r for r in results if r.expiry == expiry]
        w = sum(r.win for r in subset)
        print(f"  {expiry:>2} min: {w}/{len(subset)} = {pct(w, len(subset))}")

    print("\nDirection breakdown:")
    for direction in ("CALL", "PUT"):
        subset = [r for r in results if r.direction == direction]
        if subset:
            w = sum(r.win for r in subset)
            print(f"  {direction:4s}: {w}/{len(subset)} = {pct(w, len(subset))}")

    print("\nPair breakdown:")
    for pair in sorted(set(r.pair for r in results)):
        subset = [r for r in results if r.pair == pair]
        w = sum(r.win for r in subset)
        print(f"  {pair:7s} {w:>2}/{len(subset):<2} = {pct(w, len(subset))}")

    print("\nScore bands:")
    for low, high in ((76, 79), (80, 84), (85, 89), (90, 100)):
        subset = [r for r in results if low <= r.score <= high]
        if subset:
            w = sum(r.win for r in subset)
            print(f"  {low}-{high}: {w}/{len(subset)} = {pct(w, len(subset))}")

    print("\nIMPORTANT:")
    print("- Quality index is NOT a probability of winning.")
    print("- A historical backtest cannot guarantee future results.")
    print("- V2 should be validated on a fresh period before any use, even in demo.")


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
            symbol = pair
            market = "REGULAR"
            test = fetch_history(iq, symbol, 60, 180)
            if len(test) < strategy.MIN_CANDLES:
                otc = discover_otc(iq, pair)
                if otc:
                    symbol = otc
                    market = "OTC"
                else:
                    print(f"{pair}: no usable regular/OTC symbol")
                    continue

            print(f"\nTesting {pair} -> {symbol} ({market})...")
            try:
                results = run_pair(iq, pair, symbol, market)
                all_results.extend(results)
                if results:
                    w = sum(r.win for r in results)
                    print(f"  Result: {w}/{len(results)} wins ({pct(w, len(results))})")
                else:
                    print("  Result: no qualifying V2 signals")
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
