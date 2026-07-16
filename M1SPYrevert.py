#!/usr/bin/env python3
"""
M1SPYrevert -- SPY daily mean-reversion strategy with walk-forward train/validate.

Strategy family
---------------
Long-only mean reversion on SPY daily bars (Larry Connors style). SPY has a
strong upward drift, so shorting oversold-bounces is a poor bet; we only take
long entries when price is stretched *below* its short-term mean and then exit
on reversion.

Signals (all computed on adjusted/close price):
  * RSI(rsi_period)          -- short-period RSI, classic Connors "RSI-2".
  * z-score(zwin)            -- (close - SMA) / rolling std, distance from mean.
  * SMA(trend_win)           -- long trend filter; optionally only buy in uptrend.

Entry (go long) when ALL enabled conditions hold at the close:
  * RSI(rsi_period)     <  rsi_buy
  * z-score             <  z_buy            (i.e. stretched below the mean)
  * close               >  SMA(trend_win)   (trend filter; can be disabled)

Exit (flatten) when ANY holds:
  * RSI(rsi_period)     >  rsi_exit
  * z-score             >= z_exit           (reverted back to the mean)
  * holding_days        >= max_hold

Train / validate
----------------
The sample is split chronologically: the first `train_years` (default 2) are
used to grid-search the parameters that maximise the training objective
(default: Sharpe). The chosen parameters are then evaluated, unchanged, on the
final `validate_years` (default 1) -- a genuine out-of-sample check.

Data
----
`load_prices()` auto-detects Interactive Brokers / TWS CSV exports as well as
generic OHLCV CSVs. Point --data at a single CSV or at a directory such as
    C:\\TWS\\EquityHistoricalData
and it will locate the SPY file inside.

Usage
-----
    python M1SPYrevert.py --data C:\\TWS\\EquityHistoricalData
    python M1SPYrevert.py --data data/SPY_daily.csv --symbol SPY
    python M1SPYrevert.py --data ... --no-trend-filter --plot equity.png

Dependencies: pandas, numpy (matplotlib only if --plot is used).
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Data loading (TWS/IB export aware)
# --------------------------------------------------------------------------- #
def _find_symbol_file(directory: str, symbol: str) -> str:
    """Return the most likely CSV for `symbol` inside a directory."""
    sym = symbol.upper()
    candidates = []
    for path in glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True):
        base = os.path.basename(path).upper()
        if sym in base:
            candidates.append(path)
    if not candidates:
        # fall back to any csv so the user gets a clear error downstream
        candidates = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found for symbol {symbol!r} under {directory!r}"
        )
    # prefer the shortest basename (e.g. SPY.csv over SPY_options_2023.csv)
    candidates.sort(key=lambda p: (len(os.path.basename(p)), p))
    return candidates[0]


# columns names we accept, mapped to canonical names (all compared lower-case)
_COLUMN_ALIASES = {
    "date": "date", "datetime": "date", "time": "date", "timestamp": "date",
    "day": "date",
    "open": "open", "o": "open",
    "high": "high", "h": "high",
    "low": "low", "l": "low",
    "close": "close", "c": "close", "last": "close",
    "adj close": "close", "adjclose": "close", "adjusted close": "close",
    "wap": "close",  # IB weighted-average price fallback
    "volume": "volume", "vol": "volume", "v": "volume",
}


def _parse_dates(col: pd.Series) -> pd.Series:
    """Parse a date column robustly across the formats TWS/IB and generic CSVs
    use: compact 'YYYYMMDD' (which pandas would otherwise read as an int and
    misinterpret as epoch-nanoseconds), 'YYYY-MM-DD', with or without a time,
    and IB's 'YYYYMMDD  HH:MM:SS'."""
    s = col.astype(str).str.strip()
    # Strip an intraday time component if present (keep the date part).
    s = s.str.replace(r"[T\s]+\d{1,2}:\d{2}(:\d{2})?.*$", "", regex=True)
    # Compact 8-digit dates -> explicit format.
    compact = s.str.fullmatch(r"\d{8}")
    if compact.fillna(False).mean() > 0.8:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def load_prices(path: str, symbol: str = "SPY") -> pd.DataFrame:
    """
    Load an OHLCV daily series from a CSV file or a directory of CSVs.

    Handles Interactive Brokers / TWS exports (which may be headerless with the
    layout Date,Open,High,Low,Close,Volume[,...]) as well as generic OHLCV CSVs
    with a header row in any common column order/naming.

    Returns a DataFrame indexed by DatetimeIndex with at least a 'close' column
    (plus open/high/low/volume when available), sorted ascending, de-duplicated.
    """
    if os.path.isdir(path):
        path = _find_symbol_file(path, symbol)

    # Peek at the first non-empty line to decide whether there is a header.
    with open(path, "r", newline="") as fh:
        first_line = ""
        for line in fh:
            if line.strip():
                first_line = line.strip()
                break
    # Treat the file as having a header unless the 2nd field is numeric-looking
    # (a headerless OHLCV row starts with a date then numbers).
    fields = first_line.split(",")
    def _looks_numeric(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False
    has_header = not (_looks_numeric(fields[1]) if len(fields) > 1 else False)

    if has_header:
        df = pd.read_csv(path)
        df = df.rename(columns={c: _COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().lower())
                                for c in df.columns})
    else:
        # Assume the canonical IB/TWS export order.
        ncols = len(fields)
        names = ["date", "open", "high", "low", "close", "volume"][:ncols]
        if ncols < 5:
            raise ValueError(
                f"{path!r} looks headerless but has only {ncols} columns; "
                "expected at least Date,Open,High,Low,Close."
            )
        df = pd.read_csv(path, header=None, names=names + [f"extra{i}" for i in range(ncols - len(names))])

    if "close" not in df.columns:
        raise ValueError(
            f"Could not find a close/last price column in {path!r}. "
            f"Columns seen: {list(df.columns)}"
        )
    if "date" not in df.columns:
        # fall back to the first column
        df = df.rename(columns={df.columns[0]: "date"})

    df["date"] = _parse_dates(df["date"])
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["close"])
    if df.empty:
        raise ValueError(f"No usable rows after parsing {path!r}.")
    return df


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0, 100.0)  # no losses -> RSI 100
    return out


def zscore(close: pd.Series, window: int) -> pd.Series:
    sma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    return (close - sma) / sd.replace(0.0, np.nan)


# --------------------------------------------------------------------------- #
# Parameters + backtest
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    rsi_period: int = 2
    rsi_buy: float = 10.0
    rsi_exit: float = 65.0
    zwin: int = 20
    z_buy: float = -1.5
    z_exit: float = 0.0
    trend_win: int = 200
    use_trend_filter: bool = True
    use_rsi: bool = True
    use_z: bool = True
    max_hold: int = 10
    cost_bps: float = 1.0  # per-side cost in basis points (commission+slippage)


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    trades: int
    metrics: dict = field(default_factory=dict)


def _signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["ret"] = close.pct_change()
    out["rsi"] = rsi(close, p.rsi_period) if p.use_rsi else np.nan
    out["z"] = zscore(close, p.zwin) if p.use_z else np.nan
    out["trend_sma"] = close.rolling(p.trend_win).mean() if p.use_trend_filter else np.nan
    return out


def backtest(df: pd.DataFrame, p: Params) -> BacktestResult:
    """
    Event-style long/flat backtest. Signals are computed on the close of day t;
    the resulting position is held over day t+1's return (no look-ahead).
    Transaction cost `cost_bps` is charged on every change in position.
    """
    s = _signals(df, p)
    n = len(s)
    pos = np.zeros(n)
    holding = 0
    in_pos = False

    rsi_v = s["rsi"].to_numpy()
    z_v = s["z"].to_numpy()
    close_v = s["close"].to_numpy()
    trend_v = s["trend_sma"].to_numpy()

    for i in range(n):
        if in_pos:
            holding += 1
            exit_signal = False
            if p.use_rsi and not np.isnan(rsi_v[i]) and rsi_v[i] > p.rsi_exit:
                exit_signal = True
            if p.use_z and not np.isnan(z_v[i]) and z_v[i] >= p.z_exit:
                exit_signal = True
            if holding >= p.max_hold:
                exit_signal = True
            if exit_signal:
                in_pos = False
                holding = 0
        else:
            entry = True
            if p.use_rsi:
                entry &= (not np.isnan(rsi_v[i])) and rsi_v[i] < p.rsi_buy
            if p.use_z:
                entry &= (not np.isnan(z_v[i])) and z_v[i] < p.z_buy
            if p.use_trend_filter:
                entry &= (not np.isnan(trend_v[i])) and close_v[i] > trend_v[i]
            if not (p.use_rsi or p.use_z):
                entry = False  # guard: need at least one entry signal
            if entry:
                in_pos = True
                holding = 0
        pos[i] = 1.0 if in_pos else 0.0

    position = pd.Series(pos, index=s.index, name="position")
    # position decided at close of t is earned on t+1's return
    strat_ret = position.shift(1).fillna(0.0) * s["ret"].fillna(0.0)

    # transaction costs on position changes
    turns = position.diff().abs().fillna(position.abs())
    cost = turns * (p.cost_bps / 1e4)
    strat_ret = strat_ret - cost

    equity = (1.0 + strat_ret).cumprod()
    trades = int((position.diff() > 0).sum())
    res = BacktestResult(equity=equity, returns=strat_ret, position=position, trades=trades)
    res.metrics = compute_metrics(strat_ret, position, trades)
    return res


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(returns: pd.Series, position: pd.Series, trades: int) -> dict:
    returns = returns.fillna(0.0)
    n = len(returns)
    if n == 0:
        return {}
    total_return = float((1.0 + returns).prod() - 1.0)
    years = n / TRADING_DAYS
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 and (1.0 + total_return) > 0 else float("nan")
    vol = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))
    mean_ann = float(returns.mean() * TRADING_DAYS)
    sharpe = float(mean_ann / vol) if vol > 0 else 0.0
    # Standard target downside deviation (target = 0): RMS of the negative part
    # over ALL periods, so flat/positive days correctly contribute zero.
    downside_sq = np.minimum(returns.to_numpy(), 0.0) ** 2
    dvol = float(np.sqrt(downside_sq.mean()) * np.sqrt(TRADING_DAYS))
    sortino = float(mean_ann / dvol) if dvol > 0 else 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    exposure = float((position != 0).mean())
    # per-trade stats
    active = returns[position.shift(1).fillna(0.0) != 0.0]
    win_rate = float((active > 0).mean()) if len(active) else float("nan")
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "exposure": exposure,
        "win_rate_daily": win_rate,
        "trades": trades,
        "days": n,
    }


def buy_and_hold_metrics(df: pd.DataFrame) -> dict:
    ret = df["close"].pct_change().fillna(0.0)
    pos = pd.Series(1.0, index=df.index)
    return compute_metrics(ret, pos, trades=1)


# --------------------------------------------------------------------------- #
# Train / validate
# --------------------------------------------------------------------------- #
DEFAULT_GRID = {
    "rsi_period": [2, 3, 4],
    "rsi_buy": [5.0, 10.0, 15.0, 20.0],
    "rsi_exit": [55.0, 65.0, 75.0],
    "zwin": [20],
    "z_buy": [-1.0, -1.5, -2.0, -2.5],
    "z_exit": [-0.25, 0.0, 0.5],
    "max_hold": [5, 10],
}


def split_train_validate(df: pd.DataFrame, train_years: float, validate_years: float):
    """Chronological split: last `validate_years` are validation, the
    `train_years` immediately before that are training."""
    end = df.index.max()
    val_start = end - pd.DateOffset(days=int(round(validate_years * 365.25)))
    train_start = val_start - pd.DateOffset(days=int(round(train_years * 365.25)))
    train = df[(df.index > train_start) & (df.index <= val_start)]
    valid = df[df.index > val_start]
    return train, valid, train_start, val_start


OBJECTIVES = {
    "sharpe": lambda m: m.get("sharpe", float("-inf")),
    "cagr": lambda m: m.get("cagr", float("-inf")),
    "calmar": lambda m: m.get("calmar", float("-inf")) if m.get("calmar") == m.get("calmar") else float("-inf"),
    "total_return": lambda m: m.get("total_return", float("-inf")),
}


def grid_search(train: pd.DataFrame, base: Params, grid: dict, objective: str,
                min_trades: int = 5, verbose: bool = False):
    """Exhaustive grid search on the training window. Returns (best_params,
    best_metrics, leaderboard)."""
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    scorer = OBJECTIVES[objective]
    results = []
    for combo in combos:
        kwargs = asdict(base)
        kwargs.update(dict(zip(keys, combo)))
        p = Params(**kwargs)
        res = backtest(train, p)
        m = res.metrics
        if m.get("trades", 0) < min_trades:
            continue
        score = scorer(m)
        if score != score:  # nan
            continue
        results.append((score, kwargs, m))
    if not results:
        raise RuntimeError(
            "Grid search produced no candidate with the minimum number of "
            f"trades ({min_trades}). Try loosening the grid or --min-trades."
        )
    results.sort(key=lambda r: r[0], reverse=True)
    if verbose:
        print(f"  evaluated {len(combos)} combos, {len(results)} met the "
              f"min-trades={min_trades} filter")
    best_score, best_kwargs, best_m = results[0]
    return Params(**best_kwargs), best_m, results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_metrics(m: dict) -> str:
    if not m:
        return "  (no data)"
    def pct(x):
        return "n/a" if x != x else f"{x*100:6.2f}%"
    return (
        f"  total return : {pct(m.get('total_return'))}\n"
        f"  CAGR         : {pct(m.get('cagr'))}\n"
        f"  ann. vol     : {pct(m.get('ann_vol'))}\n"
        f"  Sharpe       : {m.get('sharpe', float('nan')):6.2f}\n"
        f"  Sortino      : {m.get('sortino', float('nan')):6.2f}\n"
        f"  max drawdown : {pct(m.get('max_drawdown'))}\n"
        f"  Calmar       : {m.get('calmar', float('nan')):6.2f}\n"
        f"  exposure     : {pct(m.get('exposure'))}\n"
        f"  daily win %  : {pct(m.get('win_rate_daily'))}\n"
        f"  trades       : {m.get('trades', 0)}\n"
        f"  days         : {m.get('days', 0)}"
    )


def print_report(train, valid, best: Params, train_m, valid_m,
                 train_start, val_start, objective):
    line = "=" * 66
    print(line)
    print("M1SPYrevert -- SPY daily mean-reversion  (train / validate report)")
    print(line)
    print(f"Train window   : {train.index.min().date()} -> {train.index.max().date()}  "
          f"({len(train)} bars)")
    print(f"Validate window: {valid.index.min().date()} -> {valid.index.max().date()}  "
          f"({len(valid)} bars)")
    print(f"Objective      : maximise {objective} on the training window")
    print()
    print("Best parameters (chosen on TRAIN only):")
    for k, v in asdict(best).items():
        print(f"    {k:16s} = {v}")
    print()
    print("IN-SAMPLE  (train):")
    print(_fmt_metrics(train_m))
    print()
    print("OUT-OF-SAMPLE  (validate):")
    print(_fmt_metrics(valid_m))
    print()
    bh = buy_and_hold_metrics(valid)
    print("Buy & hold SPY over the validation window (benchmark):")
    print(_fmt_metrics(bh))
    print(line)


def maybe_plot(valid: pd.DataFrame, res: BacktestResult, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plot skipped: matplotlib unavailable: {e}]", file=sys.stderr)
        return
    bh = (1.0 + valid["close"].pct_change().fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(res.equity.index, res.equity.values, label="M1SPYrevert (strategy)")
    ax.plot(bh.index, bh.values, label="Buy & hold SPY", alpha=0.7)
    ax.set_title("M1SPYrevert -- out-of-sample equity curve")
    ax.set_ylabel("growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[saved plot -> {path}]")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SPY daily mean-reversion train/validate backtester.")
    ap.add_argument("--data", required=True,
                    help="CSV file OR directory (e.g. C:\\TWS\\EquityHistoricalData).")
    ap.add_argument("--symbol", default="SPY", help="Symbol to locate inside a directory.")
    ap.add_argument("--train-years", type=float, default=2.0)
    ap.add_argument("--validate-years", type=float, default=1.0)
    ap.add_argument("--objective", choices=list(OBJECTIVES), default="sharpe")
    ap.add_argument("--min-trades", type=int, default=5,
                    help="Discard train candidates with fewer trades than this.")
    ap.add_argument("--cost-bps", type=float, default=1.0,
                    help="Per-side transaction cost in basis points.")
    ap.add_argument("--no-trend-filter", action="store_true",
                    help="Disable the SMA(trend_win) uptrend filter.")
    ap.add_argument("--plot", metavar="PNG", default=None,
                    help="Write an out-of-sample equity-curve PNG to this path.")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    df = load_prices(args.data, symbol=args.symbol)
    span_years = (df.index.max() - df.index.min()).days / 365.25
    need = args.train_years + args.validate_years
    print(f"Loaded {len(df)} bars for {args.symbol}: "
          f"{df.index.min().date()} -> {df.index.max().date()} ({span_years:.2f} yrs)")
    if span_years + 0.05 < need:
        print(f"WARNING: only {span_years:.2f} yrs of data but train+validate "
              f"needs {need:.2f} yrs. Results will be truncated.", file=sys.stderr)

    train, valid, train_start, val_start = split_train_validate(
        df, args.train_years, args.validate_years)
    if len(train) < 60 or len(valid) < 20:
        print(f"ERROR: not enough data after split (train={len(train)}, "
              f"validate={len(valid)} bars).", file=sys.stderr)
        return 2

    base = Params(cost_bps=args.cost_bps, use_trend_filter=not args.no_trend_filter)
    grid = {k: v for k, v in DEFAULT_GRID.items()}
    if args.verbose:
        print("Running grid search on the training window...")
    best, train_m, _ = grid_search(
        train, base, grid, args.objective,
        min_trades=args.min_trades, verbose=args.verbose)

    # Evaluate the frozen params out-of-sample.
    valid_res = backtest(valid, best)
    valid_m = valid_res.metrics

    print_report(train, valid, best, train_m, valid_m, train_start, val_start, args.objective)

    if args.plot:
        maybe_plot(valid, valid_res, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
