"""Bar-by-bar simulation and the performance metrics.

Fill model, deliberately pessimistic:
  * Entry is at the NEXT bar's open after the trigger bar closes. The 9 EMA
    cross is only known once that bar has closed, so entering on the trigger
    bar's close would be look-ahead.
  * Stops and targets are checked against each bar's high/low.
  * If a bar's range spans both the stop and the target, the STOP is assumed
    to fill first. Without tick data there is no way to know the path, and
    assuming otherwise is the single easiest way to manufacture a fake edge.
  * Slippage and commission are charged on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .indicators import build_features
from .strategy import (
    Config,
    Signal,
    generate_signals,
    initial_stop,
    initial_target,
    passes_filters,
    _band_columns,
)

SHARES_PER_CONTRACT = 100  # 1 "contract" = 1 round lot = 100 shares


@dataclass
class Trade:
    session: pd.Timestamp
    direction: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    stop: float
    target: Optional[float]
    reason: str
    bars_held: int
    gross_points: float
    net_points: float
    mae: float   # max adverse excursion, points
    mfe: float   # max favourable excursion, points


def _simulate_trade(feat: pd.DataFrame, sig: Signal, cfg: Config) -> Optional[Trade]:
    n = len(feat)
    entry_bar = sig.entry_index + 1  # fill on the next bar's open
    if entry_bar >= n:
        return None

    d = sig.direction
    entry_price = float(feat["open"].iloc[entry_bar])

    stop = initial_stop(feat, sig, cfg, entry_price)
    target = initial_target(feat, sig, cfg, entry_price, stop)

    # A stop on the wrong side of entry means the setup is already invalid.
    if (d == 1 and stop >= entry_price) or (d == -1 and stop <= entry_price):
        return None
    if target is not None:
        if (d == 1 and target <= entry_price) or (d == -1 and target >= entry_price):
            return None

    risk = abs(entry_price - stop)
    up_col, dn_col = _band_columns(cfg.dev_band)

    highs = feat["high"].to_numpy()
    lows = feat["low"].to_numpy()
    closes = feat["close"].to_numpy()
    atrs = feat["atr"].to_numpy()
    vwaps = feat["vwap"].to_numpy()

    mae = 0.0
    mfe = 0.0
    moved_to_be = False
    exit_price = None
    exit_bar = None
    reason = ""

    last_bar = n - 1
    for j in range(entry_bar, n):
        hi, lo = highs[j], lows[j]

        fav = (hi - entry_price) if d == 1 else (entry_price - lo)
        adv = (entry_price - lo) if d == 1 else (hi - entry_price)
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # --- stop first (pessimistic tie-break) ---------------------------
        hit_stop = lo <= stop if d == 1 else hi >= stop
        if hit_stop:
            exit_price, exit_bar, reason = stop, j, "stop"
            break

        # --- target -------------------------------------------------------
        if target is not None:
            hit_target = hi >= target if d == 1 else lo <= target
            if hit_target:
                exit_price, exit_bar, reason = target, j, "target"
                break

        # --- breakeven ratchet -------------------------------------------
        if cfg.breakeven_at_r is not None and not moved_to_be:
            if fav >= cfg.breakeven_at_r * risk:
                stop = entry_price
                moved_to_be = True

        # --- ATR trail ----------------------------------------------------
        if cfg.trail_atr_mult is not None and j > entry_bar:
            trail = (
                closes[j] - cfg.trail_atr_mult * atrs[j]
                if d == 1
                else closes[j] + cfg.trail_atr_mult * atrs[j]
            )
            stop = max(stop, trail) if d == 1 else min(stop, trail)

        # --- time stop ----------------------------------------------------
        if cfg.time_stop_bars is not None and (j - entry_bar) >= cfg.time_stop_bars:
            exit_price, exit_bar, reason = closes[j], j, "time"
            break

        # --- session close ------------------------------------------------
        if j == last_bar and cfg.exit_at_close:
            exit_price, exit_bar, reason = closes[j], j, "close"
            break

    if exit_price is None:
        exit_price, exit_bar, reason = closes[last_bar], last_bar, "close"

    gross = d * (exit_price - entry_price)
    costs = 2.0 * (cfg.slippage_points + cfg.commission_points)
    net = gross - costs

    return Trade(
        session=feat.index[entry_bar].normalize(),
        direction=d,
        entry_time=feat.index[entry_bar],
        entry_price=float(entry_price),
        exit_time=feat.index[exit_bar],
        exit_price=float(exit_price),
        stop=float(stop),
        target=float(target) if target is not None else None,
        reason=reason,
        bars_held=int(exit_bar - entry_bar),
        gross_points=float(gross),
        net_points=float(net),
        mae=float(mae),
        mfe=float(mfe),
    )


def run(feat_by_session: dict, cfg: Config) -> pd.DataFrame:
    """Run one config across pre-computed per-session feature frames."""
    trades: list[Trade] = []

    for _, feat in feat_by_session.items():
        if len(feat) < 10:
            continue
        taken = 0
        for sig in generate_signals(feat, cfg):
            if cfg.max_trades_per_session is not None and taken >= cfg.max_trades_per_session:
                break
            if not passes_filters(feat, sig, cfg):
                continue
            tr = _simulate_trade(feat, sig, cfg)
            if tr is not None:
                trades.append(tr)
                taken += 1

    if not trades:
        return pd.DataFrame(
            columns=[
                "session", "direction", "entry_time", "entry_price", "exit_time",
                "exit_price", "stop", "target", "reason", "bars_held",
                "gross_points", "net_points", "mae", "mfe",
            ]
        )
    return pd.DataFrame([t.__dict__ for t in trades]).sort_values("entry_time").reset_index(drop=True)


def prepare_sessions(df: pd.DataFrame, cfg: Config) -> dict:
    """Build features once per session so a variant sweep does not recompute
    indicators for every config."""
    feat = build_features(df, ema_span=cfg.ema_span)
    return {day: g for day, g in feat.groupby("session") if len(g) >= 10}


def metrics(trades: pd.DataFrame, n_sessions: int) -> dict:
    """The scorecard: win%, profit factor, total points, points per contract,
    and drawdown measured on the trade-by-trade equity curve."""
    if trades.empty:
        return {
            "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "total_points": 0.0, "points_per_contract": 0.0,
            "avg_points": 0.0, "max_drawdown_points": 0.0,
            "max_drawdown_per_contract": 0.0, "expectancy_r": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "largest_loss": 0.0,
            "sessions": n_sessions, "trades_per_session": 0.0,
            "longs": 0, "shorts": 0, "sharpe": 0.0,
        }

    pnl = trades["net_points"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    max_dd = float(-drawdown.min())

    total_points = float(pnl.sum())
    risk = (trades["entry_price"] - trades["stop"]).abs()
    r_multiples = pnl / risk.replace(0.0, np.nan)

    std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = (float(pnl.mean()) / std * np.sqrt(252.0)) if std > 1e-9 else 0.0

    # Round once, then scale, so the printed per-contract figures reconcile
    # exactly with the printed point figures.
    total_points_r = round(total_points, 2)
    max_dd_r = round(max_dd, 2)

    return {
        "trades": int(len(trades)),
        "win_rate": round(100.0 * len(wins) / len(pnl), 2),
        "profit_factor": round(pf, 3) if np.isfinite(pf) else float("inf"),
        "total_points": total_points_r,
        "points_per_contract": round(total_points_r * SHARES_PER_CONTRACT, 2),
        "avg_points": round(float(pnl.mean()), 4),
        "max_drawdown_points": max_dd_r,
        "max_drawdown_per_contract": round(max_dd_r * SHARES_PER_CONTRACT, 2),
        "expectancy_r": round(float(r_multiples.mean()), 3) if r_multiples.notna().any() else 0.0,
        "avg_win": round(float(wins.mean()), 3) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 3) if len(losses) else 0.0,
        "largest_loss": round(float(pnl.min()), 3),
        "sessions": n_sessions,
        "trades_per_session": round(len(pnl) / max(n_sessions, 1), 2),
        "longs": int((trades["direction"] == 1).sum()),
        "shorts": int((trades["direction"] == -1).sum()),
        "sharpe": round(sharpe, 2),
    }
