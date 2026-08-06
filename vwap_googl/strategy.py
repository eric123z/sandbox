"""The VWAP 2-sigma / 9-EMA cross strategy and its tunable variants.

Core setup (the user's specification, unchanged):
    1. Price action touches one of the 2nd standard deviation VWAP bands.
    2. The 9 EMA then crosses back over the VWAP centreline.
    3. That cross is the entry.

A lower-band touch arms a LONG (fade the flush, enter when the short-term
average reclaims VWAP); an upper-band touch arms a SHORT. Everything beyond
those three rules lives in Config so variants can be swept and compared
without touching the signal logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd

StopMode = Literal["band", "atr", "touch_extreme"]
TargetMode = Literal["vwap", "opposite_band", "r_multiple", "atr"]


@dataclass
class Config:
    """One fully specified variant of the strategy."""

    name: str = "base"

    # --- core setup -------------------------------------------------------
    dev_band: float = 2.0            # which sigma band arms the trade
    ema_span: int = 9
    arm_bars: int = 12               # cross must occur within N bars of touch
    allow_long: bool = True
    allow_short: bool = True

    # --- exits ------------------------------------------------------------
    stop_mode: StopMode = "touch_extreme"
    stop_atr_mult: float = 1.5       # used when stop_mode == "atr"
    stop_band_pad_atr: float = 0.25  # cushion beyond the band / extreme
    target_mode: TargetMode = "vwap"
    target_r: float = 2.0            # used when target_mode == "r_multiple"
    target_atr_mult: float = 2.0     # used when target_mode == "atr"
    trail_atr_mult: Optional[float] = None
    breakeven_at_r: Optional[float] = None
    time_stop_bars: Optional[int] = None
    exit_at_close: bool = True

    # --- filters (None = disabled) ---------------------------------------
    entry_window: tuple[int, int] = (0, 390)   # minutes from the 09:30 open
    min_rvol: Optional[float] = None
    max_adx: Optional[float] = None
    min_adx: Optional[float] = None
    max_abs_vwap_slope: Optional[float] = None
    min_band_width: Optional[float] = None
    max_band_width: Optional[float] = None
    long_rsi_max: Optional[float] = None       # long only if RSI below this
    short_rsi_min: Optional[float] = None      # short only if RSI above this
    max_touch_number: Optional[int] = None     # e.g. 1 = first touch of day only
    max_trades_per_session: Optional[int] = None
    require_vwap_reclaim: bool = False         # price, not just EMA, back over VWAP

    # --- costs ------------------------------------------------------------
    slippage_points: float = 0.02              # per side, in dollars/share
    commission_points: float = 0.005           # per side, in dollars/share

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Signal:
    entry_index: int          # bar index at which the trigger completed
    direction: int            # +1 long, -1 short
    touch_index: int
    touch_extreme: float      # session extreme reached during the armed window
    touch_number: int


def _band_columns(dev: float) -> tuple[str, str]:
    tag = str(dev).rstrip("0").rstrip(".")
    return f"vwap_up{tag}", f"vwap_dn{tag}"


def generate_signals(feat: pd.DataFrame, cfg: Config) -> list[Signal]:
    """Walk one session's bars and emit every trigger the config allows.

    State machine per direction: idle -> armed (band touched) -> fired (EMA
    crossed VWAP within arm_bars). The arm decays so a touch at 10:00 cannot
    justify an entry at 15:00.
    """
    up_col, dn_col = _band_columns(cfg.dev_band)
    if up_col not in feat.columns:
        raise KeyError(f"Band column {up_col} not built; check dev_band={cfg.dev_band}")

    high = feat["high"].to_numpy()
    low = feat["low"].to_numpy()
    close = feat["close"].to_numpy()
    upper = feat[up_col].to_numpy()
    lower = feat[dn_col].to_numpy()
    vwap = feat["vwap"].to_numpy()
    ema9 = feat["ema9"].to_numpy()
    diff = ema9 - vwap

    signals: list[Signal] = []
    n = len(feat)

    # Armed state: bar index of the touch, and the extreme price since then.
    long_arm: Optional[int] = None
    long_extreme = np.nan
    short_arm: Optional[int] = None
    short_extreme = np.nan
    long_touches = 0
    short_touches = 0

    for i in range(1, n):
        if np.isnan(vwap[i]) or np.isnan(ema9[i]) or np.isnan(upper[i]):
            continue

        # --- arming: did price reach the band? ---------------------------
        if low[i] <= lower[i]:
            if long_arm is None:
                long_touches += 1
                long_extreme = low[i]
            else:
                long_extreme = min(long_extreme, low[i])
            long_arm = i
        if high[i] >= upper[i]:
            if short_arm is None:
                short_touches += 1
                short_extreme = high[i]
            else:
                short_extreme = max(short_extreme, high[i])
            short_arm = i

        # --- decay stale arms --------------------------------------------
        if long_arm is not None and i - long_arm > cfg.arm_bars:
            long_arm, long_extreme = None, np.nan
        if short_arm is not None and i - short_arm > cfg.arm_bars:
            short_arm, short_extreme = None, np.nan

        # --- trigger: 9 EMA crosses the VWAP centreline -------------------
        crossed_up = diff[i - 1] <= 0 < diff[i]
        crossed_down = diff[i - 1] >= 0 > diff[i]

        if long_arm is not None and crossed_up and cfg.allow_long:
            if not cfg.require_vwap_reclaim or close[i] > vwap[i]:
                signals.append(
                    Signal(
                        entry_index=i,
                        direction=1,
                        touch_index=long_arm,
                        touch_extreme=long_extreme,
                        touch_number=long_touches,
                    )
                )
                long_arm, long_extreme = None, np.nan

        if short_arm is not None and crossed_down and cfg.allow_short:
            if not cfg.require_vwap_reclaim or close[i] < vwap[i]:
                signals.append(
                    Signal(
                        entry_index=i,
                        direction=-1,
                        touch_index=short_arm,
                        touch_extreme=short_extreme,
                        touch_number=short_touches,
                    )
                )
                short_arm, short_extreme = None, np.nan

    return signals


def passes_filters(feat: pd.DataFrame, sig: Signal, cfg: Config) -> bool:
    """Regime and context gates evaluated at the trigger bar.

    Only information available at that bar is read -- no forward peeking.
    """
    row = feat.iloc[sig.entry_index]

    mins = row["min_from_open"]
    lo, hi = cfg.entry_window
    if not (lo <= mins <= hi):
        return False

    if cfg.min_rvol is not None and row["rvol"] < cfg.min_rvol:
        return False
    if cfg.max_adx is not None and row["adx"] > cfg.max_adx:
        return False
    if cfg.min_adx is not None and row["adx"] < cfg.min_adx:
        return False
    if cfg.max_abs_vwap_slope is not None and abs(row["vwap_slope"]) > cfg.max_abs_vwap_slope:
        return False
    if cfg.min_band_width is not None and row["band_width"] < cfg.min_band_width:
        return False
    if cfg.max_band_width is not None and row["band_width"] > cfg.max_band_width:
        return False
    if cfg.max_touch_number is not None and sig.touch_number > cfg.max_touch_number:
        return False

    if sig.direction == 1 and cfg.long_rsi_max is not None and row["rsi"] > cfg.long_rsi_max:
        return False
    if sig.direction == -1 and cfg.short_rsi_min is not None and row["rsi"] < cfg.short_rsi_min:
        return False

    return True


def initial_stop(feat: pd.DataFrame, sig: Signal, cfg: Config, entry_price: float) -> float:
    """Where the idea is wrong. Anchored beyond the extreme that armed the
    trade -- if price takes out the flush low, the fade has failed."""
    row = feat.iloc[sig.entry_index]
    atr_val = row["atr"]
    pad = cfg.stop_band_pad_atr * atr_val

    if cfg.stop_mode == "atr":
        return entry_price - sig.direction * cfg.stop_atr_mult * atr_val

    if cfg.stop_mode == "band":
        up_col, dn_col = _band_columns(cfg.dev_band)
        band = row[dn_col] if sig.direction == 1 else row[up_col]
        return band - sig.direction * pad

    # touch_extreme (default)
    return sig.touch_extreme - sig.direction * pad


def initial_target(
    feat: pd.DataFrame, sig: Signal, cfg: Config, entry_price: float, stop: float
) -> Optional[float]:
    row = feat.iloc[sig.entry_index]
    risk = abs(entry_price - stop)

    if cfg.target_mode == "vwap":
        # The mean-reversion objective: price returns to fair value.
        # Entry already sits near VWAP, so push to the opposite 1-sigma.
        sigma = row["vwap_sigma"]
        return row["vwap"] + sig.direction * sigma

    if cfg.target_mode == "opposite_band":
        up_col, dn_col = _band_columns(cfg.dev_band)
        return row[up_col] if sig.direction == 1 else row[dn_col]

    if cfg.target_mode == "r_multiple":
        return entry_price + sig.direction * cfg.target_r * risk

    if cfg.target_mode == "atr":
        return entry_price + sig.direction * cfg.target_atr_mult * row["atr"]

    return None
