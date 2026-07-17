#!/usr/bin/env python3
"""Generate clearly-synthetic daily OHLC sample data for pipeline demos.

These series are random walks with drift — NOT real market prices. They exist
only so the WOLF pipeline can be demonstrated where market-data downloads are
blocked.
"""

import csv
import datetime as dt
import os

import numpy as np

SPECS = {  # start price, annualized drift, annualized vol, rng seed
    "SAMPLE_LYFT": (14.0, 0.10, 0.55, 11),
    "SAMPLE_PLTR": (25.0, 0.45, 0.60, 22),
    "SAMPLE_MSFT": (330.0, 0.20, 0.25, 33),
}
YEARS = 3


def make(name, spec, outdir):
    p0, mu, sigma, seed = spec
    rng = np.random.default_rng(seed)
    n = 252 * YEARS
    daily_mu, daily_sig = mu / 252, sigma / np.sqrt(252)
    rets = rng.normal(daily_mu, daily_sig, n)
    close = p0 * np.exp(np.cumsum(rets))
    o = np.empty(n)
    o[0] = p0
    o[1:] = close[:-1] * np.exp(rng.normal(0, daily_sig / 4, n - 1))
    spread = np.abs(rng.normal(0, daily_sig, n)) * close
    h = np.maximum(o, close) + spread / 2
    l = np.minimum(o, close) - spread / 2

    day = dt.date(2023, 7, 17)
    rows = []
    i = 0
    while len(rows) < n:
        if day.weekday() < 5:
            rows.append((day.isoformat(), round(o[i], 2), round(h[i], 2),
                         round(l[i], 2), round(close[i], 2)))
            i += 1
        day += dt.timedelta(days=1)

    path = os.path.join(outdir, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Open", "High", "Low", "Close"])
        w.writerows(rows)
    print(f"wrote {path} ({n} rows, synthetic)")


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(outdir, exist_ok=True)
    for name, spec in SPECS.items():
        make(name, spec, outdir)
