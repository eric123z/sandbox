# M1SPYrevert — SPY daily mean-reversion (train / validate)

A long-only, Larry-Connors-style mean-reversion strategy for SPY daily bars,
with a walk-forward **2-year train / 1-year validate** split and a parameter
grid-search on the training window only.

## Install

```bash
pip install -r requirements.txt
```

## Run on your own data

The loader auto-detects Interactive Brokers / TWS CSV exports (headerless
`Date,Open,High,Low,Close,Volume` with compact `YYYYMMDD` dates) as well as
generic OHLCV CSVs with a header row. Point `--data` at the file or the folder:

```bash
# Windows / your desktop:
python M1SPYrevert.py --data "C:\TWS\EquityHistoricalData" --symbol SPY --plot equity.png

# a single CSV:
python M1SPYrevert.py --data data/SPY_daily.csv
```

If `--data` is a directory, it finds the CSV whose name contains the `--symbol`
(e.g. `SPY.csv`, `SPY_1day.csv`).

## What it does

1. Loads the daily series and splits it chronologically: the **last 1 year** is
   held out for validation; the **2 years** before that are the training set.
2. **Grid-searches** the strategy parameters on the *training* window to
   maximise the chosen objective (`--objective sharpe|cagr|calmar|total_return`).
3. **Freezes** the best parameters and evaluates them, unchanged, on the
   held-out validation year — a genuine out-of-sample test — alongside a
   buy-&-hold SPY benchmark.

### Signals

| Indicator | Role | Default |
|---|---|---|
| RSI(`rsi_period`) | short-period oversold trigger (Connors "RSI-2") | period 2 |
| z-score(`zwin`) | distance below the rolling mean | 20-day window |
| SMA(`trend_win`) | long-trend filter — only buy in an uptrend | 200-day |

**Entry (long)** when RSI < `rsi_buy` **and** z-score < `z_buy` **and** (if the
filter is on) close > SMA(`trend_win`).
**Exit** when RSI > `rsi_exit`, or z-score reverts to ≥ `z_exit`, or the
position has been held `max_hold` days.

Signals are computed at the close of day *t* and the position is earned on day
*t+1*'s return (no look-ahead). A per-side transaction cost (`--cost-bps`,
default 1 bp) is charged on every position change.

## Key flags

| Flag | Meaning |
|---|---|
| `--train-years` / `--validate-years` | change the split (default 2 / 1) |
| `--objective` | training objective: `sharpe` (default), `cagr`, `calmar`, `total_return` |
| `--no-trend-filter` | drop the SMA-200 uptrend filter |
| `--cost-bps` | per-side cost in basis points |
| `--min-trades` | discard training candidates with too few trades |
| `--plot equity.png` | write an out-of-sample equity-curve chart |

## Reproducing the smoke test (no real data)

`scripts/make_synthetic_spy.py` writes a **synthetic, not-real** SPY-like series
(TWS-style CSV) purely to exercise the code path. Numbers from it are
meaningless — replace with your real TWS export for actual results.

```bash
python scripts/make_synthetic_spy.py --out data/SYNTHETIC_SPY_daily.csv
python M1SPYrevert.py --data data/SYNTHETIC_SPY_daily.csv --plot data/synthetic_equity.png
```

## Notes on interpretation

- The strategy is **in the market only a small fraction of the time** (it waits
  for dips), so expect low exposure, low volatility, and fewer trades than
  buy-&-hold. Compare risk-adjusted numbers (Sharpe/Sortino/Calmar), not just
  total return.
- A big gap between in-sample and out-of-sample results is the grid-search
  overfitting — exactly what the validation split is there to reveal. Widen the
  data, shrink the grid, or raise `--min-trades` if you see it.
