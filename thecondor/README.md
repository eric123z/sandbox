# TheCondor

An automated iron condor trader for the **SPY** ETF on
[Alpaca](https://alpaca.markets), built around Alpaca's multi-leg (MLEG)
order support so all four legs fill together at one net credit — no legging
risk.

## Strategy

Each run of `trade` builds one condor:

1. Fetch the SPY option chain 7–14 days to expiration (configurable).
2. Sell the put and call whose deltas are closest to ±0.15 (configurable).
3. Buy wings 5 points further out (configurable).
4. Submit all four legs as a single MLEG limit order at the mid-price net
   credit minus a small concession. Alpaca's convention: a **negative**
   limit price on an MLEG order means a net credit.

## Assignment — read this part

SPY options are **American-style and physically settled**. A short leg
that finishes in the money is assigned 100 SPY shares per contract, and
early assignment is possible (mainly deep-ITM calls right before SPY's
quarterly ex-dividend date). TheCondor manages this instead of holding
through settlement:

- `manage` **always closes the position on expiration day** (or earlier
  with `--close-dte N`), regardless of profit — run it daily during market
  hours.
- `manage` also takes profit early once half the credit has decayed
  (configurable with `--take-profit`).
- The expiration check reads the position symbols directly, so the safety
  close works even if the local state file is lost.

## Setup

```bash
pip install -r requirements.txt
export APCA_API_KEY_ID=...       # or ALPACA_API_KEY
export APCA_API_SECRET_KEY=...   # or ALPACA_SECRET_KEY
```

The script defaults to Alpaca's **paper** endpoint. Pass `--live` to trade
a live account — spreads require options trading level 3 on Alpaca.

## Usage

```bash
python thecondor.py trade --dry-run          # print the plan without ordering
python thecondor.py trade                    # open 1 condor
python thecondor.py trade --qty 2 --short-delta 0.20 --width 10 --min-dte 5 --max-dte 10
python thecondor.py status                   # positions, open orders, saved state
python thecondor.py manage                   # take profit / forced expiration-day close
python thecondor.py manage --take-profit 0.4 --close-dte 1
python thecondor.py close                    # buy the condor back now
```

`trade` refuses to stack a second condor while SPY option positions are
open, and aborts if the net credit is below `--min-credit` (default 0.50).
Fill details are saved to `thecondor_state.json` so `manage` knows the
credit received.

To automate it, schedule `trade` weekly and `manage` daily (cron, GitHub
Actions, etc.) during market hours.

## Risk

An iron condor's loss is capped but large relative to its credit: a 5-wide
condor collecting 1.00 risks 4.00 per contract, plus assignment risk on
American-style options if positions are not closed before expiration. This
code is an example for paper trading, not investment advice.
