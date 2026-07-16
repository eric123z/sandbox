#!/usr/bin/env python3
"""
Generate a SYNTHETIC SPY-like daily series for smoke-testing M1SPYrevert.py.

*** THIS IS NOT REAL MARKET DATA. ***

It is a geometric random walk with drift, occasional mean-reverting shocks and
a few drawdowns, written in an Interactive-Brokers / TWS style CSV so the
loader's format auto-detection gets exercised. Use it only to verify the code
runs end-to-end; performance numbers from it are meaningless.
"""
import argparse
import numpy as np
import pandas as pd


def generate(start="2023-06-01", days=780, seed=42, s0=420.0):
    rng = np.random.default_rng(seed)
    # business-day calendar
    idx = pd.bdate_range(start=start, periods=days)
    n = len(idx)
    mu = 0.08 / 252          # ~8% annual drift
    sigma = 0.012            # daily vol
    # AR(1) mean-reverting component to create tradeable dips
    ar = np.zeros(n)
    phi = 0.85
    shocks = rng.normal(0, 0.006, n)
    for i in range(1, n):
        ar[i] = phi * ar[i - 1] + shocks[i]
    rets = mu + rng.normal(0, sigma, n) - 0.3 * ar
    # inject a couple of sharp drawdowns + recoveries
    for center in (int(n * 0.3), int(n * 0.65)):
        for k in range(6):
            if center + k < n:
                rets[center + k] -= 0.015
        for k in range(6, 16):
            if center + k < n:
                rets[center + k] += 0.010
    price = s0 * np.exp(np.cumsum(rets))
    high = price * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[s0], price[:-1]])
    vol = rng.integers(50_000_000, 120_000_000, n)
    df = pd.DataFrame({
        "Date": idx.strftime("%Y%m%d"),   # TWS-style compact date, no header dtype hint
        "Open": np.round(open_, 2),
        "High": np.round(high, 2),
        "Low": np.round(low, 2),
        "Close": np.round(price, 2),
        "Volume": vol,
    })
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/SYNTHETIC_SPY_daily.csv")
    ap.add_argument("--days", type=int, default=780)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = generate(days=args.days, seed=args.seed)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out} "
          f"({df['Date'].iloc[0]} .. {df['Date'].iloc[-1]})  [SYNTHETIC / NOT REAL]")
