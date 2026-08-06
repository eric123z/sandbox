"""The candidate improvements to test on top of the base setup.

Each variant changes ONE idea (or one coherent bundle) relative to base so the
attribution is readable. The list is intentionally modest in size: sweeping
thousands of combinations over 120 sessions guarantees an overfit winner.
"""

from __future__ import annotations

from dataclasses import replace

from .strategy import Config

BASE = Config(name="base")


def core_variants() -> list[Config]:
    """One-factor-at-a-time tests against the base configuration."""
    v: list[Config] = [BASE]

    # --- exit structure ---------------------------------------------------
    v += [
        replace(BASE, name="exit_vwap_touch", target_mode="vwap"),
        replace(BASE, name="exit_opposite_band", target_mode="opposite_band"),
        replace(BASE, name="exit_2R", target_mode="r_multiple", target_r=2.0),
        replace(BASE, name="exit_1R", target_mode="r_multiple", target_r=1.0),
        replace(BASE, name="exit_1.5R", target_mode="r_multiple", target_r=1.5),
        replace(BASE, name="exit_atr2", target_mode="atr", target_atr_mult=2.0),
    ]

    # --- stop placement ---------------------------------------------------
    v += [
        replace(BASE, name="stop_atr1.0", stop_mode="atr", stop_atr_mult=1.0),
        replace(BASE, name="stop_atr1.5", stop_mode="atr", stop_atr_mult=1.5),
        replace(BASE, name="stop_atr2.5", stop_mode="atr", stop_atr_mult=2.5),
        replace(BASE, name="stop_band", stop_mode="band"),
        replace(BASE, name="stop_extreme_wide", stop_band_pad_atr=0.5),
    ]

    # --- trade management -------------------------------------------------
    v += [
        replace(BASE, name="trail_atr2", trail_atr_mult=2.0),
        replace(BASE, name="trail_atr3", trail_atr_mult=3.0),
        replace(BASE, name="breakeven_1R", breakeven_at_r=1.0),
        replace(BASE, name="breakeven_0.5R", breakeven_at_r=0.5),
        replace(BASE, name="time_stop_12", time_stop_bars=12),
        replace(BASE, name="time_stop_24", time_stop_bars=24),
    ]

    # --- time-of-day ------------------------------------------------------
    # The open is a trend-drive window where fading 2-sigma is how accounts
    # die; the midday balance is where mean reversion actually pays.
    v += [
        replace(BASE, name="tod_skip_first_30", entry_window=(30, 390)),
        replace(BASE, name="tod_skip_first_60", entry_window=(60, 390)),
        replace(BASE, name="tod_no_last_30", entry_window=(0, 360)),
        replace(BASE, name="tod_core", entry_window=(30, 360)),
        replace(BASE, name="tod_midday_only", entry_window=(60, 330)),
    ]

    # --- regime filters ---------------------------------------------------
    v += [
        replace(BASE, name="adx_ranging_25", max_adx=25.0),
        replace(BASE, name="adx_ranging_20", max_adx=20.0),
        replace(BASE, name="flat_vwap_only", max_abs_vwap_slope=0.0015),
        replace(BASE, name="flat_vwap_tight", max_abs_vwap_slope=0.0008),
        replace(BASE, name="rvol_min_1.0", min_rvol=1.0),
        replace(BASE, name="rvol_min_1.5", min_rvol=1.5),
        replace(BASE, name="wide_bands_only", min_band_width=0.004),
    ]

    # --- setup quality ----------------------------------------------------
    v += [
        replace(BASE, name="first_touch_only", max_touch_number=1),
        replace(BASE, name="first_two_touches", max_touch_number=2),
        replace(BASE, name="require_price_reclaim", require_vwap_reclaim=True),
        replace(BASE, name="one_trade_per_day", max_trades_per_session=1),
        replace(BASE, name="arm_6", arm_bars=6),
        replace(BASE, name="arm_20", arm_bars=20),
    ]

    # --- direction --------------------------------------------------------
    v += [
        replace(BASE, name="longs_only", allow_short=False),
        replace(BASE, name="shorts_only", allow_long=False),
    ]

    # --- band selection ---------------------------------------------------
    v += [
        replace(BASE, name="band_3sigma", dev_band=3.0),
        replace(BASE, name="band_1sigma", dev_band=1.0),
    ]

    return v


def combo_variants(winners: list[str]) -> list[Config]:
    """Stack the individually strongest ideas.

    Kept small and hand-built rather than a full cross-product: with ~120
    in-sample sessions, a large grid finds noise, not edge.
    """
    combos: list[Config] = []

    combos.append(
        replace(
            BASE,
            name="combo_regime",
            entry_window=(30, 360),
            max_adx=25.0,
            max_abs_vwap_slope=0.0015,
        )
    )
    combos.append(
        replace(
            BASE,
            name="combo_regime_quality",
            entry_window=(30, 360),
            max_adx=25.0,
            max_abs_vwap_slope=0.0015,
            max_touch_number=2,
            require_vwap_reclaim=True,
        )
    )
    combos.append(
        replace(
            BASE,
            name="combo_regime_managed",
            entry_window=(30, 360),
            max_adx=25.0,
            max_abs_vwap_slope=0.0015,
            breakeven_at_r=1.0,
            target_mode="vwap",
        )
    )
    combos.append(
        replace(
            BASE,
            name="combo_tight_risk",
            entry_window=(30, 360),
            max_adx=25.0,
            stop_mode="atr",
            stop_atr_mult=1.5,
            target_mode="r_multiple",
            target_r=1.5,
            breakeven_at_r=0.75,
        )
    )
    combos.append(
        replace(
            BASE,
            name="combo_selective",
            entry_window=(60, 330),
            max_adx=20.0,
            max_abs_vwap_slope=0.0008,
            min_rvol=1.0,
            max_touch_number=1,
            max_trades_per_session=1,
            breakeven_at_r=1.0,
        )
    )
    combos.append(
        replace(
            BASE,
            name="combo_trend_ride",
            entry_window=(30, 360),
            max_adx=25.0,
            trail_atr_mult=2.0,
            target_mode="opposite_band",
        )
    )
    return combos
