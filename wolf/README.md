# WOLF — Wide Optimization of Long/Flat strategies

WOLF grid-searches ~250 parameter combinations across seven families of daily
technical strategies (SMA cross, EMA cross, RSI mean reversion, Donchian
breakout, MACD, Bollinger mean reversion, momentum) and ranks them by total
points captured. It then renders a readable `.odt` report with, per stock:

- **Win%** — winning trades ÷ total trades
- **IF** — impact factor (profit factor): gross winning points ÷ gross losing points
- **Max DD** — worst peak-to-trough equity drop, in points
- **Total Points** — points captured across all trades (1 point = $1/share)
- **Points/Trade** — total points ÷ trades

Backtest model: long/flat, one unit, signals on the daily close, close-to-close
accounting, no costs unless `--cost` is given. Strategies with fewer than 8
trades are discarded so a single lucky trade can't win the leaderboard.

## Produce the real report (needs internet access to stooq.com)

```bash
pip install numpy
python3 wolf.py --download LYFT,PLTR,MSFT --datadir data --outdir results
python3 make_report.py results/LYFT.json results/PLTR.json results/MSFT.json \
    --out WOLF_report.odt
```

Or with your own CSVs (`Date,Open,High,Low,Close`, daily rows):

```bash
python3 wolf.py --csv data/LYFT.csv --ticker LYFT --outdir results
```

The `.odt` writer emits OpenDocument XML directly — no LibreOffice or odfpy
needed.

## Demo on synthetic data (no network needed)

```bash
python3 make_sample_data.py
python3 wolf.py --csv data/SAMPLE_LYFT.csv --ticker SAMPLE_LYFT --outdir results
python3 wolf.py --csv data/SAMPLE_PLTR.csv --ticker SAMPLE_PLTR --outdir results
python3 wolf.py --csv data/SAMPLE_MSFT.csv --ticker SAMPLE_MSFT --outdir results
python3 make_report.py results/SAMPLE_*.json --out WOLF_sample_report.odt --sample
```

`--sample` stamps a red banner on the report so synthetic output can never be
mistaken for real market results.

## Caveats

The winners are *in-sample* optima — the metrics are optimistically biased by
the optimization itself. Validate out-of-sample (e.g. walk-forward) before
trading anything. Not investment advice.
