#!/usr/bin/env python3
"""WOLF — Wide Optimization of Long/Flat strategies.

Grid-searches a zoo of classic long/flat technical strategies over daily
OHLC data and ranks parameter sets by total points captured.

Metrics reported per strategy:
  Win%      — winning trades / total trades
  IF        — impact factor (profit factor): gross winning points / gross losing points
  Max DD    — deepest peak-to-trough drop of the strategy equity curve, in points
  Total Pts — sum of points captured across all trades (1 point = $1/share)
  Pts/Trade — Total Pts / number of trades

Execution model: signals are computed on the daily close; position changes
take effect at that same close (close-to-close accounting). Long/flat only,
one unit, no costs — add costs via --cost if desired.

Usage:
  python3 wolf.py --csv data/LYFT.csv --ticker LYFT --out results/LYFT.json
  python3 wolf.py --download LYFT,PLTR,MSFT --outdir results   (needs internet)
"""

import argparse
import csv
import io
import json
import math
import os
import sys
import urllib.request

import numpy as np

MIN_TRADES = 8  # parameter sets with fewer trades are ignored (degenerate winners)


# ---------------------------------------------------------------- data ------

def load_csv(path):
    """Load a daily OHLC CSV (stooq or yahoo format). Returns dict of arrays."""
    dates, o, h, l, c = [], [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {k.lower().strip(): k for k in reader.fieldnames}
        need = ["date", "open", "high", "low", "close"]
        if not all(k in cols for k in need):
            raise SystemExit(f"{path}: need columns {need}, got {reader.fieldnames}")
        for row in reader:
            try:
                vals = [float(row[cols[k]]) for k in need[1:]]
            except (ValueError, TypeError):
                continue  # skip null/dividend rows
            dates.append(row[cols["date"]])
            o.append(vals[0]); h.append(vals[1]); l.append(vals[2]); c.append(vals[3])
    if len(c) < 260:
        raise SystemExit(f"{path}: only {len(c)} usable rows; need at least a year of data")
    return {
        "date": np.array(dates),
        "open": np.array(o), "high": np.array(h),
        "low": np.array(l), "close": np.array(c),
    }


def download(ticker, dest):
    """Fetch daily history from stooq (no key needed). Works only where the
    network allows it — the Claude sandbox network policy may block this."""
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode()
    if not text.startswith("Date,"):
        raise RuntimeError(f"unexpected response for {ticker}: {text[:80]}")
    with open(dest, "w") as f:
        f.write(text)
    return dest


# ----------------------------------------------------------- indicators -----

def sma(x, n):
    out = np.full(len(x), np.nan)
    if n <= len(x):
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def ema(x, n):
    out = np.empty(len(x))
    alpha = 2.0 / (n + 1)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(close, n):
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    # Wilder smoothing
    au = np.empty(len(close)); ad = np.empty(len(close))
    au[0] = up[:n].mean() if len(close) > n else 0
    ad[0] = dn[:n].mean() if len(close) > n else 0
    k = 1.0 / n
    for i in range(1, len(close)):
        au[i] = au[i - 1] + k * (up[i] - au[i - 1])
        ad[i] = ad[i - 1] + k * (dn[i] - ad[i - 1])
    rs = np.divide(au, ad, out=np.full(len(close), np.inf), where=ad != 0)
    return 100 - 100 / (1 + rs)


def rolling_max(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = x[i - n + 1:i + 1].max()
    return out


def rolling_min(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = x[i - n + 1:i + 1].min()
    return out


def rolling_std(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = x[i - n + 1:i + 1].std(ddof=0)
    return out


# ----------------------------------------------------------- strategies -----
# Each generator yields (name, params_dict, position_array) where position is
# 1 (long) or 0 (flat) decided on each bar's close.

def strat_ma_cross(d, kind):
    close = d["close"]
    f_ma = {n: (sma if kind == "SMA" else ema)(close, n) for n in (5, 10, 15, 20, 30, 50)}
    s_ma = {n: (sma if kind == "SMA" else ema)(close, n) for n in (20, 30, 50, 100, 150, 200)}
    for f in (5, 10, 15, 20, 30, 50):
        for s in (20, 30, 50, 100, 150, 200):
            if f >= s:
                continue
            pos = (f_ma[f] > s_ma[s]).astype(float)
            pos[np.isnan(f_ma[f]) | np.isnan(s_ma[s])] = 0
            yield f"{kind} cross", {"fast": f, "slow": s}, pos


def strat_rsi(d):
    close = d["close"]
    for n in (2, 3, 4, 7, 14):
        r = rsi(close, n)
        for buy in (10, 15, 20, 25, 30):
            for sell in (50, 55, 60, 65, 70, 80):
                pos = np.zeros(len(close))
                holding = 0.0
                for i in range(n, len(close)):
                    if holding == 0 and r[i] < buy:
                        holding = 1.0
                    elif holding == 1 and r[i] > sell:
                        holding = 0.0
                    pos[i] = holding
                yield "RSI mean reversion", {"period": n, "buy_below": buy, "sell_above": sell}, pos


def strat_donchian(d):
    close, high, low = d["close"], d["high"], d["low"]
    for n_in in (10, 20, 30, 55, 100):
        hi = rolling_max(high, n_in)
        for n_out in (5, 10, 20, 50):
            if n_out >= n_in:
                continue
            lo = rolling_min(low, n_out)
            pos = np.zeros(len(close))
            holding = 0.0
            for i in range(1, len(close)):
                if np.isnan(hi[i - 1]) or np.isnan(lo[i - 1]):
                    pos[i] = holding
                    continue
                if holding == 0 and close[i] > hi[i - 1]:
                    holding = 1.0
                elif holding == 1 and close[i] < lo[i - 1]:
                    holding = 0.0
                pos[i] = holding
            yield "Donchian breakout", {"entry": n_in, "exit": n_out}, pos


def strat_macd(d):
    close = d["close"]
    for f in (8, 12):
        for s in (21, 26, 35):
            for sig in (5, 9):
                line = ema(close, f) - ema(close, s)
                signal = ema(line, sig)
                pos = (line > signal).astype(float)
                pos[:s] = 0
                yield "MACD", {"fast": f, "slow": s, "signal": sig}, pos


def strat_bollinger(d):
    close = d["close"]
    for n in (10, 20):
        mid = sma(close, n)
        sd = rolling_std(close, n)
        for k in (1.5, 2.0, 2.5):
            lower = mid - k * sd
            pos = np.zeros(len(close))
            holding = 0.0
            for i in range(n, len(close)):
                if holding == 0 and close[i] < lower[i]:
                    holding = 1.0
                elif holding == 1 and close[i] > mid[i]:
                    holding = 0.0
                pos[i] = holding
            yield "Bollinger mean reversion", {"period": n, "k": k}, pos


def strat_momentum(d):
    close = d["close"]
    for n in (20, 40, 60, 120, 250):
        pos = np.zeros(len(close))
        pos[n:] = (close[n:] > close[:-n]).astype(float)
        yield "Momentum (ROC>0)", {"lookback": n}, pos


def all_strategies(d):
    yield from strat_ma_cross(d, "SMA")
    yield from strat_ma_cross(d, "EMA")
    yield from strat_rsi(d)
    yield from strat_donchian(d)
    yield from strat_macd(d)
    yield from strat_bollinger(d)
    yield from strat_momentum(d)


# ------------------------------------------------------------- backtest -----

def evaluate(close, pos, cost):
    """Close-to-close long/flat backtest. Returns metrics dict or None."""
    daily = np.diff(close) * pos[:-1]          # points earned each day while long
    equity = np.concatenate([[0.0], np.cumsum(daily)])

    # split into trades: contiguous runs of pos == 1
    changes = np.diff(np.concatenate([[0.0], pos]))
    entries = np.where(changes == 1)[0]
    exits = np.where(changes == -1)[0]
    if pos[-1] == 1:                            # still long on last bar → close out
        exits = np.append(exits, len(pos) - 1)
    n_trades = len(entries)
    if n_trades < MIN_TRADES:
        return None
    trade_pts = close[exits] - close[entries] - cost
    wins = trade_pts[trade_pts > 0]
    losses = trade_pts[trade_pts <= 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    total = trade_pts.sum()
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max())
    return {
        "trades": int(n_trades),
        "win_pct": round(100.0 * len(wins) / n_trades, 1),
        "impact_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_dd_points": round(max_dd, 2),
        "total_points": round(float(total), 2),
        "points_per_trade": round(float(total) / n_trades, 2),
    }


def optimize(d, ticker, cost):
    close = d["close"]
    results = []
    for name, params, pos in all_strategies(d):
        m = evaluate(close, pos, cost)
        if m is None:
            continue
        m["strategy"] = name
        m["params"] = params
        results.append(m)
    results.sort(key=lambda r: r["total_points"], reverse=True)
    bh = float(close[-1] - close[0])
    return {
        "ticker": ticker,
        "bars": len(close),
        "date_range": [str(d["date"][0]), str(d["date"][-1])],
        "last_close": round(float(close[-1]), 2),
        "buy_hold_points": round(bh, 2),
        "combos_tested": sum(1 for _ in all_strategies(d)),
        "best": results[0] if results else None,
        "top": results[:5],
    }


def main():
    ap = argparse.ArgumentParser(description="WOLF strategy optimizer")
    ap.add_argument("--csv", help="path to daily OHLC csv")
    ap.add_argument("--ticker", help="ticker label for --csv")
    ap.add_argument("--download", help="comma-separated tickers to fetch from stooq")
    ap.add_argument("--datadir", default="data", help="where downloaded csvs go")
    ap.add_argument("--outdir", default="results", help="where result json goes")
    ap.add_argument("--cost", type=float, default=0.0, help="round-trip cost in points per trade")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    jobs = []
    if args.download:
        os.makedirs(args.datadir, exist_ok=True)
        for t in args.download.split(","):
            t = t.strip().upper()
            path = os.path.join(args.datadir, f"{t}.csv")
            print(f"downloading {t} → {path}")
            download(t, path)
            jobs.append((t, path))
    elif args.csv:
        jobs.append(((args.ticker or os.path.basename(args.csv).split(".")[0]).upper(), args.csv))
    else:
        ap.error("give --csv or --download")

    for ticker, path in jobs:
        d = load_csv(path)
        res = optimize(d, ticker, args.cost)
        out = os.path.join(args.outdir, f"{ticker}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        b = res["best"]
        print(f"{ticker}: {res['bars']} bars, {res['combos_tested']} combos → "
              f"best {b['strategy']} {b['params']} | win% {b['win_pct']} IF {b['impact_factor']} "
              f"DD {b['max_dd_points']} total {b['total_points']} pts/trade {b['points_per_trade']}"
              if b else f"{ticker}: no strategy met the {MIN_TRADES}-trade minimum")

if __name__ == "__main__":
    main()
