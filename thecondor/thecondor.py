#!/usr/bin/env python3
"""
TheCondor — an automated SPY iron condor trader for Alpaca.

Sells a delta-targeted iron condor on the SPY ETF as a single atomic
multi-leg (MLEG) order:

    sell 1 OTM put   (short put,  ~target delta)
    buy  1 further OTM put   (put wing,  short strike - width)
    sell 1 OTM call  (short call, ~target delta)
    buy  1 further OTM call  (call wing, short strike + width)

SPY options are American-style and physically settled: a short leg that
finishes in the money is assigned 100 shares per contract, and early
assignment is possible (mainly deep-ITM calls before an ex-dividend date).
Because of that, `manage` closes the condor on expiration day (or
`--close-dte` days before) instead of holding through settlement, and can
also take profit early once a fraction of the credit has decayed away.

Commands:
    trade    Build and submit a new condor (skips if one is already open).
    manage   Take profit if the target is hit; always close by expiration.
    status   Show open SPY option positions, open orders, and saved state.
    close    Buy back the open condor now at the current mid price.

Credentials are read from the environment:
    APCA_API_KEY_ID / APCA_API_SECRET_KEY  (or ALPACA_API_KEY / ALPACA_SECRET_KEY)

The script defaults to Alpaca's paper endpoint; pass --live to trade a live
account (requires options trading level 3 for spreads).

This is example code, not investment advice. Iron condors have limited
profit and can lose the full spread width minus the credit received.
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, OptionLegRequest

UNDERLYING = "SPY"
STATE_FILE = "thecondor_state.json"
# SPY is in the penny interval program: options quote in $0.01 increments.
def tick_for(price: float) -> float:
    return 0.01

OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


@dataclass
class Contract:
    symbol: str
    kind: str          # "C" or "P"
    strike: float
    expiration: date
    bid: float
    ask: float
    delta: float | None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


def parse_occ(symbol: str):
    m = OCC_RE.match(symbol)
    if not m:
        return None
    root, ymd, kind, strike = m.groups()
    return root, datetime.strptime(ymd, "%y%m%d").date(), kind, int(strike) / 1000


def get_keys() -> tuple[str, str]:
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the environment.")
    return key, secret


def load_state() -> dict | None:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def save_state(state: dict | None) -> None:
    if state is None:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
    else:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)


def open_condor_symbols(trading: TradingClient) -> list:
    """Open SPY option positions, as (symbol, signed_qty) pairs."""
    out = []
    for pos in trading.get_all_positions():
        parsed = parse_occ(pos.symbol)
        if parsed and parsed[0] == UNDERLYING:
            out.append((pos.symbol, float(pos.qty)))
    return out


def fetch_chain(data: OptionHistoricalDataClient, min_dte: int, max_dte: int) -> list[Contract]:
    today = date.today()
    chain = data.get_option_chain(
        OptionChainRequest(
            underlying_symbol=UNDERLYING,
            expiration_date_gte=today + timedelta(days=min_dte),
            expiration_date_lte=today + timedelta(days=max_dte),
        )
    )
    contracts = []
    for symbol, snap in chain.items():
        parsed = parse_occ(symbol)
        quote = getattr(snap, "latest_quote", None)
        if not parsed or quote is None:
            continue
        bid, ask = float(quote.bid_price or 0), float(quote.ask_price or 0)
        if ask <= 0:
            continue
        greeks = getattr(snap, "greeks", None)
        delta = float(greeks.delta) if greeks and greeks.delta is not None else None
        contracts.append(Contract(symbol, parsed[2], parsed[3], parsed[1], bid, ask, delta))
    if not contracts:
        sys.exit(f"No quotable {UNDERLYING} contracts found {min_dte}-{max_dte} DTE. "
                 "Check your options data subscription/feed.")
    return contracts


def nearest_by_delta(contracts: list[Contract], target: float) -> Contract:
    candidates = [c for c in contracts if c.delta is not None
                  and 0.02 <= abs(c.delta) <= 0.40]
    if not candidates:
        sys.exit("No contracts with usable deltas near the target — try a different DTE window.")
    return min(candidates, key=lambda c: abs(abs(c.delta) - abs(target)))


def wing_for(contracts: list[Contract], short: Contract, width: float) -> Contract:
    desired = short.strike + width if short.kind == "C" else short.strike - width
    beyond = [c for c in contracts
              if (c.strike > short.strike if short.kind == "C" else c.strike < short.strike)]
    if not beyond:
        sys.exit(f"No wing strike available beyond the short {short.kind} at {short.strike}.")
    return min(beyond, key=lambda c: abs(c.strike - desired))


def estimate_underlying(calls: list[Contract], puts: list[Contract]) -> float | None:
    """Rough spot estimate: the strike where call and put mids are closest (ATM)."""
    put_by_strike = {p.strike: p for p in puts}
    best, best_diff = None, float("inf")
    for c in calls:
        p = put_by_strike.get(c.strike)
        if p:
            diff = abs(c.mid - p.mid)
            if diff < best_diff:
                best, best_diff = c.strike, diff
    return best


def floor_to_tick(price: float) -> float:
    tick = tick_for(price)
    return round(math.floor(round(price / tick, 6)) * tick, 2)


def ceil_to_tick(price: float) -> float:
    tick = tick_for(price)
    return round(math.ceil(round(price / tick, 6)) * tick, 2)


def cmd_trade(args, trading: TradingClient, data: OptionHistoricalDataClient) -> None:
    existing = open_condor_symbols(trading)
    if existing:
        print(f"A {UNDERLYING} options position is already open — not stacking another condor:")
        for sym, qty in existing:
            print(f"  {sym:>21}  qty {qty:+g}")
        return

    contracts = fetch_chain(data, args.min_dte, args.max_dte)
    expiration = min({c.expiration for c in contracts})
    contracts = [c for c in contracts if c.expiration == expiration]
    calls = sorted((c for c in contracts if c.kind == "C"), key=lambda c: c.strike)
    puts = sorted((c for c in contracts if c.kind == "P"), key=lambda c: c.strike)

    short_call = nearest_by_delta(calls, +args.short_delta)
    short_put = nearest_by_delta(puts, -args.short_delta)
    if short_put.strike >= short_call.strike:
        sys.exit(f"Inverted condor (short put {short_put.strike} >= short call "
                 f"{short_call.strike}) — widen the delta target or DTE window.")
    long_call = wing_for(calls, short_call, args.width)
    long_put = wing_for(puts, short_put, args.width)

    credit = short_call.mid + short_put.mid - long_call.mid - long_put.mid
    limit_credit = floor_to_tick(credit - args.slippage)
    width = max(long_call.strike - short_call.strike, short_put.strike - long_put.strike)
    collateral = (width - limit_credit) * 100
    spot = estimate_underlying(calls, puts)

    print(f"TheCondor — {UNDERLYING} iron condor, expiring {expiration} "
          f"({(expiration - date.today()).days} DTE)"
          + (f", spot ~{spot:g}" if spot else ""))
    for label, c, side in (("long put ", long_put, "BUY "), ("short put", short_put, "SELL"),
                           ("short call", short_call, "SELL"), ("long call", long_call, "BUY ")):
        d = f" delta {c.delta:+.3f}" if c.delta is not None else ""
        print(f"  {side} {label:<10} {c.symbol:>21}  strike {c.strike:>7g}  "
              f"mid {c.mid:6.2f}{d}")
    print(f"  net credit (mid) {credit:.2f} -> limit {limit_credit:.2f}, "
          f"width {width:g}, collateral ~${collateral:,.0f}/condor, "
          f"credit/collateral {limit_credit * 100 / collateral:.1%}, qty {args.qty}")

    if limit_credit < args.min_credit:
        sys.exit(f"Credit {limit_credit:.2f} is below the minimum {args.min_credit:.2f} — not trading.")
    if args.dry_run:
        print("Dry run — no order submitted.")
        return

    order = trading.submit_order(LimitOrderRequest(
        qty=args.qty,
        limit_price=-limit_credit,  # negative = net credit for MLEG orders
        order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.DAY,
        legs=[
            OptionLegRequest(symbol=long_put.symbol, ratio_qty=1,
                             side=OrderSide.BUY, position_intent=PositionIntent.BUY_TO_OPEN),
            OptionLegRequest(symbol=short_put.symbol, ratio_qty=1,
                             side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=short_call.symbol, ratio_qty=1,
                             side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=long_call.symbol, ratio_qty=1,
                             side=OrderSide.BUY, position_intent=PositionIntent.BUY_TO_OPEN),
        ],
    ))
    print(f"Submitted order {order.id} ({order.status}).")
    save_state({
        "order_id": str(order.id),
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "expiration": expiration.isoformat(),
        "credit": limit_credit,
        "qty": args.qty,
        "width": width,
        "legs": {"long_put": long_put.symbol, "short_put": short_put.symbol,
                 "short_call": short_call.symbol, "long_call": long_call.symbol},
    })
    print(f"State saved to {STATE_FILE}. SPY options are physically settled — "
          "run `manage` daily; it takes profit at the target and always closes "
          "by expiration to avoid assignment.")


def current_close_cost(data: OptionHistoricalDataClient, positions: list) -> float:
    """Net debit to close: pay mid on shorts, receive mid on longs."""
    symbols = [sym for sym, _ in positions]
    quotes = data.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=symbols))
    cost = 0.0
    for sym, qty in positions:
        q = quotes[sym]
        mid = (float(q.bid_price) + float(q.ask_price)) / 2
        cost += mid if qty < 0 else -mid
    return cost


def close_positions(args, trading: TradingClient, data: OptionHistoricalDataClient,
                    positions: list) -> None:
    cost = current_close_cost(data, positions)
    limit_debit = ceil_to_tick(max(cost, 0) + args.slippage)
    qty = int(min(abs(q) for _, q in positions))
    legs = [OptionLegRequest(
                symbol=sym, ratio_qty=1,
                side=OrderSide.BUY if q < 0 else OrderSide.SELL,
                position_intent=PositionIntent.BUY_TO_CLOSE if q < 0
                                else PositionIntent.SELL_TO_CLOSE)
            for sym, q in positions]
    order = trading.submit_order(LimitOrderRequest(
        qty=qty,
        limit_price=limit_debit,  # positive = net debit
        order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.DAY,
        legs=legs,
    ))
    print(f"Closing order {order.id} submitted: {qty} condor(s) at a "
          f"{limit_debit:.2f} debit (mid {cost:.2f}).")


def cmd_manage(args, trading: TradingClient, data: OptionHistoricalDataClient) -> None:
    state = load_state()
    positions = open_condor_symbols(trading)
    if not positions:
        print(f"No open {UNDERLYING} option positions.")
        save_state(None)
        return
    # Expiration comes from the position symbols, so the safety close works
    # even if the state file has been lost.
    expiration = min(parse_occ(sym)[1] for sym, _ in positions)
    dte = (expiration - date.today()).days
    if dte <= args.close_dte:
        print(f"Condor expires {expiration} ({dte} DTE) — closing to avoid assignment.")
        close_positions(args, trading, data, positions)
        save_state(None)
        return
    if not state:
        sys.exit(f"Positions are open but {STATE_FILE} is missing — take-profit "
                 "needs the credit received; use `close` to exit manually.")
    cost = current_close_cost(data, positions)
    target = state["credit"] * (1 - args.take_profit)
    print(f"Credit received {state['credit']:.2f}, cost to close now {cost:.2f}, "
          f"take-profit trigger <= {target:.2f}.")
    if cost <= target:
        close_positions(args, trading, data, positions)
        save_state(None)
    else:
        print(f"Target not reached — holding ({dte} DTE; forced close at "
              f"{args.close_dte} DTE).")


def cmd_close(args, trading: TradingClient, data: OptionHistoricalDataClient) -> None:
    positions = open_condor_symbols(trading)
    if not positions:
        print(f"No open {UNDERLYING} option positions to close.")
        save_state(None)
        return
    close_positions(args, trading, data, positions)
    save_state(None)


def cmd_status(args, trading: TradingClient, data: OptionHistoricalDataClient) -> None:
    positions = open_condor_symbols(trading)
    print(f"Open {UNDERLYING} option positions:" if positions
          else f"No open {UNDERLYING} option positions.")
    for sym, qty in positions:
        print(f"  {sym:>21}  qty {qty:+g}")
    orders = trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    open_orders = [o for o in orders
                   if any(parse_occ(leg.symbol) and parse_occ(leg.symbol)[0] == UNDERLYING
                          for leg in (o.legs or []))
                   or (parse_occ(o.symbol or "") or ("",))[0] == UNDERLYING]
    if open_orders:
        print(f"Open {UNDERLYING} orders:")
        for o in open_orders:
            print(f"  {o.id}  {o.order_class}  {o.status}  limit {o.limit_price}")
    state = load_state()
    if state:
        print(f"Saved state ({STATE_FILE}):")
        print(json.dumps(state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="TheCondor",
                                     description="Automated SPY iron condors on Alpaca.")
    parser.add_argument("--live", action="store_true",
                        help="use the live endpoint (default: paper; live spreads "
                             "require options trading level 3)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_trade = sub.add_parser("trade", help="open a new iron condor")
    p_trade.add_argument("--qty", type=int, default=1, help="number of condors (default 1)")
    p_trade.add_argument("--short-delta", type=float, default=0.15,
                         help="target |delta| for the short strikes (default 0.15)")
    p_trade.add_argument("--width", type=float, default=5.0,
                         help="wing width in index points (default 5)")
    p_trade.add_argument("--min-dte", type=int, default=7, help="minimum days to expiration")
    p_trade.add_argument("--max-dte", type=int, default=14, help="maximum days to expiration")
    p_trade.add_argument("--min-credit", type=float, default=0.50,
                         help="abort if the net credit is below this (default 0.50)")
    p_trade.add_argument("--slippage", type=float, default=0.05,
                         help="price concession off mid (default 0.05)")
    p_trade.add_argument("--dry-run", action="store_true", help="print the plan, don't submit")

    p_manage = sub.add_parser("manage",
                              help="take profit if the target is hit; always close "
                                   "by expiration to avoid assignment")
    p_manage.add_argument("--take-profit", type=float, default=0.50,
                          help="close once this fraction of the credit has been "
                               "captured (default 0.50)")
    p_manage.add_argument("--close-dte", type=int, default=0,
                          help="force-close when this many days to expiration "
                               "remain (default 0 = expiration day)")
    p_manage.add_argument("--slippage", type=float, default=0.05,
                          help="price concession past mid when closing (default 0.05)")

    p_close = sub.add_parser("close", help="close the open condor now")
    p_close.add_argument("--slippage", type=float, default=0.05,
                         help="price concession past mid when closing (default 0.05)")

    sub.add_parser("status", help="show positions, open orders, and saved state")

    args = parser.parse_args()
    key, secret = get_keys()
    trading = TradingClient(key, secret, paper=not args.live)
    data = OptionHistoricalDataClient(key, secret)

    {"trade": cmd_trade, "manage": cmd_manage,
     "close": cmd_close, "status": cmd_status}[args.command](args, trading, data)


if __name__ == "__main__":
    main()
