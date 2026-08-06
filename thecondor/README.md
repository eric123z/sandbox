# TheCondor

An automated iron condor trader for **XSP** (Mini-SPX index options) on
[Alpaca](https://alpaca.markets), built around Alpaca's multi-leg (MLEG)
order support so all four legs fill together at one net credit — no legging
risk.

## Why XSP

XSP options are **cash-settled and European-style**:

- **No assignment, ever.** An in-the-money finish debits/credits cash — you
  never receive stock, so the condor can genuinely be held to expiration.
- **No early exercise** against your short strikes.
- 1/10th the size of SPX, so a 5-point-wide condor holds roughly $500 of
  collateral per contract.

> Alpaca introduced index options (XSP, SPX, VIX, …) in **paper trading**
> in July 2026. TheCondor defaults to the paper endpoint; use `--live` only
> once Alpaca enables index options for live accounts and your account has
> options trading level 3 (required for spreads).

## Strategy

Each run of `trade` builds one condor:

1. Fetch the XSP option chain 7–14 days to expiration (configurable).
2. Sell the put and call whose deltas are closest to ±0.15 (configurable).
3. Buy wings 5 points further out (configurable).
4. Submit all four legs as a single MLEG limit order at the mid-price net
   credit minus a small concession. Alpaca's convention: a **negative**
   limit price on an MLEG order means a net credit.

The default plan is to hold to expiration and let the position cash-settle.
`manage` optionally closes early once half the credit has decayed.

## Setup

```bash
pip install -r requirements.txt
export APCA_API_KEY_ID=...       # or ALPACA_API_KEY
export APCA_API_SECRET_KEY=...   # or ALPACA_SECRET_KEY
```

## Usage

```bash
python thecondor.py trade --dry-run          # print the plan without ordering
python thecondor.py trade                    # open 1 condor
python thecondor.py trade --qty 2 --short-delta 0.20 --width 10 --min-dte 5 --max-dte 10
python thecondor.py status                   # positions, open orders, saved state
python thecondor.py manage --take-profit 0.5 # close early at 50% of max profit
python thecondor.py close                    # buy the condor back now
```

`trade` refuses to stack a second condor while XSP option positions are
open, and aborts if the net credit is below `--min-credit` (default 0.50).
Fill details are saved to `thecondor_state.json` so `manage` knows the
credit received; after a cash settlement at expiration, `manage` clears the
state automatically.

To automate it, schedule `trade` weekly and `manage` daily (cron, GitHub
Actions, etc.) during market hours.

## Risk

An iron condor's loss is capped but large relative to its credit: a 5-wide
condor collecting 1.00 risks 4.00 per contract. This code is an example for
paper trading, not investment advice.
