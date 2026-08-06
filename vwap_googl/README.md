# GOOGL — VWAP 2σ Band / 9 EMA Cross Strategy

Research harness for the setup:

> Price action touches one of the **2nd standard deviation VWAP bands**, then the
> **9 EMA crosses over the VWAP** centreline — that cross is the entry.

A lower-band touch arms a **long** (fade the flush, enter when the short-term
average reclaims VWAP). An upper-band touch arms a **short**. Everything beyond
those three rules is a tunable knob in `Config`, so variants can be compared
without touching the signal logic.

---

## Status: engine complete, **awaiting data**

The code is finished and tested. It has **not** produced real GOOGL results yet,
because no GOOGL intraday data of sufficient length is reachable from this
sandbox. See [Data](#data) below. Nothing in this repo should be read as a
backtest result until you run it against real bars.

---

## Quick start

```bash
pip install pandas numpy tzdata pytest

# 120 sessions to select the variant, next 120 to validate it
python -m vwap_googl.run --data path/to/GOOGL_5min.csv --interval 5min --train 120 --test 120
```

Accepts a file, a directory, or a glob. Column names are auto-detected across
TWS/IB exports, Yahoo dumps and generic OHLCV CSVs.

```bash
python -m pytest vwap_googl/tests/ -q     # 16 tests
```

## Data

Required: **≥240 regular-hours sessions** of GOOGL intraday bars with a real
volume field, ideally 1-min or 5-min.

Volume is not optional. VWAP is volume-weighted, so bars with zero volume
contribute nothing and silently distort the bands. `data.audit()` reports the
zero-volume percentage and `validate_for_backtest()` refuses to run above 5%
unless you pass `--force`.

**Exporting from TWS:** chart GOOGL → right-click → *Export Historical Data*.
Request RTH-only bars if the option is offered; the loader drops extended hours
regardless. IB caps a single historical request, so you will likely need several
exports — drop them all in one directory and point `--data` at the directory.

## What gets tested

`variants.py` holds ~40 single-factor variants and 6 stacked combinations:

| Group | Ideas |
|---|---|
| Exit target | VWAP touch, opposite band, 1R / 1.5R / 2R, ATR multiple |
| Stop | Beyond the touch extreme, beyond the band, ATR 1.0/1.5/2.5 |
| Management | ATR trail, breakeven ratchet, time stop |
| Time of day | Skip first 30/60 min, avoid last 30, midday only |
| Regime | ADX ceiling, flat-VWAP-slope, relative volume floor, band width |
| Setup quality | First touch only, price reclaim confirmation, one trade/day, arm window |
| Direction | Longs only, shorts only |
| Band | 1σ / 3σ instead of 2σ |

## Method

Three stages, strictly chronological:

1. **Single-factor sweep** on the first 120 sessions. One change at a time, so
   attribution is readable.
2. **Stacked combinations** of the strongest individual ideas, still in-sample.
3. **Out-of-sample validation** — the winner selected in stages 1–2 is run once
   on the following 120 sessions. Nothing is ever selected using OOS data.

Variants firing fewer than `max(20, sessions/4)` trades are disqualified: a
filter that fires four times can post a 100% win rate and mean nothing.

Ranking blends profit factor, return-per-unit-drawdown, expectancy in R, and
win rate. Win rate is weighted lightly on purpose — it is trivially inflated by
a wide stop and a tiny target, which is exactly the trade profile that blows up.

## Fill model (deliberately pessimistic)

- Entry fills at the **next bar's open** after the trigger bar closes. The EMA
  cross is only known on that close, so entering at the trigger close is
  look-ahead.
- Stops and targets are checked against each bar's high/low.
- **If a bar spans both the stop and the target, the stop fills.** Without tick
  data the path is unknowable, and assuming otherwise is the single easiest way
  to manufacture a fake edge.
- Slippage and commission are charged on both sides (defaults: $0.02 + $0.005
  per share per side).

## Metrics

`win_rate`, `profit_factor`, `total_points`, `points_per_contract`,
`max_drawdown_points`, `max_drawdown_per_contract`, `expectancy_r`, `sharpe`,
plus trade counts and long/short split.

**Points** are dollars per share (GOOGL 150.00 → 151.00 = 1.00 point).
**Per contract** assumes 1 contract = 100 shares, so points × 100. Change
`SHARES_PER_CONTRACT` in `backtest.py` if you size differently.

## Layout

```
indicators.py   Session VWAP + σ bands, session-reset 9 EMA, ATR, RSI, ADX, RVOL
data.py         Format-tolerant CSV loading, RTH filter, resampling, quality audit
strategy.py     Config, signal state machine, filters, stop/target placement
backtest.py     Trade simulation, metrics
variants.py     The candidate improvements
run.py          Three-stage study CLI
tests/          16 tests: VWAP correctness, look-ahead, fill model, metrics
```

## Known limitations

- **Overfitting risk is real.** ~46 variants against 120 sessions will produce a
  flattering in-sample winner by chance alone. The OOS stage exists to catch
  that, and the verdict line calls it out when the edge does not survive.
- Single split, not walk-forward. A rolling walk-forward would be more robust
  and is the natural next step.
- No earnings-day or halt handling. Earnings gaps are a distinct regime and
  arguably should be excluded outright.
- Assumes every signalled trade is fillable at the modelled price. Fine for
  GOOGL at 100 shares; revisit at size.
- `SHARES_PER_CONTRACT = 100` is a reporting convention, not a risk model.
  Fixed-share sizing means dollar risk varies with the stop distance.
