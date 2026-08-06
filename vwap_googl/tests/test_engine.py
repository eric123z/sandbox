"""Correctness tests. These guard the two failure modes that silently turn a
losing strategy into a winning backtest: look-ahead bias and a fill model that
resolves ambiguous bars in the strategy's favour.

Run:  python -m pytest vwap_googl/tests/ -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vwap_googl.backtest import _simulate_trade, metrics, prepare_sessions, run
from vwap_googl.indicators import build_features, ema, vwap_bands
from vwap_googl.strategy import Config, Signal, generate_signals


def make_session(prices, volumes=None, day="2024-03-01", freq="5min"):
    n = len(prices)
    idx = pd.date_range(f"{day} 09:30", periods=n, freq=freq, tz="US/Eastern")
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.10,
            "low": prices - 0.10,
            "close": prices,
            "volume": np.asarray(volumes if volumes is not None else [1000] * n, dtype=float),
        },
        index=idx,
    )


# --- VWAP correctness ----------------------------------------------------

def test_vwap_equals_manual_volume_weighted_average():
    df = make_session([100, 102, 101], volumes=[100, 200, 300])
    out = vwap_bands(df)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    expected = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    pd.testing.assert_series_equal(out["vwap"], expected, check_names=False)


def test_vwap_resets_each_session():
    d1 = make_session([100] * 5, day="2024-03-01")
    d2 = make_session([200] * 5, day="2024-03-04")
    out = vwap_bands(pd.concat([d1, d2]))
    # First bar of day two must reflect only day two's price.
    assert out["vwap"].iloc[5] == pytest.approx(200.0, abs=0.01)


def test_bands_straddle_vwap_and_widen_with_dispersion():
    df = make_session([100, 105, 95, 110, 90], volumes=[1000] * 5)
    out = vwap_bands(df)
    assert (out["vwap_up2"] >= out["vwap"]).all()
    assert (out["vwap_dn2"] <= out["vwap"]).all()
    # 2-sigma must be wider than 1-sigma wherever dispersion is non-zero.
    spread1 = (out["vwap_up1"] - out["vwap_dn1"]).iloc[-1]
    spread2 = (out["vwap_up2"] - out["vwap_dn2"]).iloc[-1]
    assert spread2 > spread1 > 0


def test_zero_volume_session_does_not_crash():
    df = make_session([100, 101, 102], volumes=[0, 0, 0])
    out = vwap_bands(df)
    assert out["vwap"].isna().all()  # undefined, not silently zero


# --- EMA -----------------------------------------------------------------

def test_ema_resets_each_session():
    d1 = make_session([100] * 10, day="2024-03-01")
    d2 = make_session([200] * 10, day="2024-03-04")
    combined = pd.concat([d1, d2])
    e = ema(combined["close"], 9)
    # Without a reset the prior day's 100 would drag this far below 200.
    assert e.iloc[10] == pytest.approx(200.0, abs=0.01)


# --- look-ahead ----------------------------------------------------------

def test_features_do_not_depend_on_future_bars():
    """Truncating the session must not change already-computed values."""
    rng = np.random.default_rng(42)
    prices = 150 + np.cumsum(rng.normal(0, 0.2, 60))
    df = make_session(prices, volumes=rng.integers(500, 5000, 60))

    full = build_features(df)
    truncated = build_features(df.iloc[:40])

    for col in ["vwap", "vwap_up2", "vwap_dn2", "ema9", "rsi", "adx", "vwap_slope"]:
        np.testing.assert_allclose(
            full[col].iloc[:40].to_numpy(),
            truncated[col].to_numpy(),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"{col} changed when future bars were removed -> look-ahead",
        )


def test_entry_fills_on_bar_after_the_trigger():
    rng = np.random.default_rng(7)
    prices = 150 + np.cumsum(rng.normal(0, 0.3, 80))
    df = make_session(prices, volumes=rng.integers(500, 5000, 80))
    feat = build_features(df)
    cfg = Config()

    sigs = generate_signals(feat, cfg)
    for sig in sigs:
        tr = _simulate_trade(feat, sig, cfg)
        if tr is None:
            continue
        # Fill must be the OPEN of the bar AFTER the signal bar.
        assert tr.entry_time == feat.index[sig.entry_index + 1]
        assert tr.entry_price == pytest.approx(feat["open"].iloc[sig.entry_index + 1])


# --- fill model ----------------------------------------------------------

def test_stop_wins_when_bar_spans_both_stop_and_target():
    """The pessimistic tie-break is the whole basis for trusting these numbers."""
    idx = pd.date_range("2024-03-01 09:30", periods=4, freq="5min", tz="US/Eastern")
    feat = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            # Bar 2 sweeps from 95 to 105: it contains both the stop and target.
            "high": [100.5, 100.5, 105.0, 100.5],
            "low": [99.5, 99.5, 95.0, 99.5],
            "close": [100.0, 100.0, 100.0, 100.0],
            "atr": [1.0] * 4,
            "vwap": [100.0] * 4,
            "vwap_sigma": [1.0] * 4,
            "min_from_open": [0.0, 5.0, 10.0, 15.0],
            "rvol": [1.0] * 4,
            "adx": [20.0] * 4,
            "rsi": [50.0] * 4,
            "vwap_slope": [0.0] * 4,
            "band_width": [0.04] * 4,
            "vwap_up2": [102.0] * 4,
            "vwap_dn2": [98.0] * 4,
        },
        index=idx,
    )
    cfg = Config(stop_mode="atr", stop_atr_mult=2.0, target_mode="r_multiple",
                 target_r=2.0, slippage_points=0.0, commission_points=0.0)
    sig = Signal(entry_index=0, direction=1, touch_index=0, touch_extreme=99.0, touch_number=1)

    tr = _simulate_trade(feat, sig, cfg)
    assert tr is not None
    assert tr.reason == "stop", "ambiguous bar must resolve against the trade"
    assert tr.net_points < 0


def test_costs_are_charged_on_both_sides():
    idx = pd.date_range("2024-03-01 09:30", periods=3, freq="5min", tz="US/Eastern")
    feat = pd.DataFrame(
        {
            "open": [100.0] * 3, "high": [100.0] * 3,
            "low": [100.0] * 3, "close": [100.0] * 3,
            "atr": [1.0] * 3, "vwap": [100.0] * 3, "vwap_sigma": [1.0] * 3,
            "min_from_open": [0.0, 5.0, 10.0], "rvol": [1.0] * 3,
            "adx": [20.0] * 3, "rsi": [50.0] * 3, "vwap_slope": [0.0] * 3,
            "band_width": [0.04] * 3,
            "vwap_up2": [102.0] * 3, "vwap_dn2": [98.0] * 3,
        },
        index=idx,
    )
    cfg = Config(stop_mode="atr", stop_atr_mult=2.0, target_mode="r_multiple",
                 target_r=2.0, slippage_points=0.02, commission_points=0.005)
    sig = Signal(entry_index=0, direction=1, touch_index=0, touch_extreme=99.0, touch_number=1)

    tr = _simulate_trade(feat, sig, cfg)
    assert tr.gross_points == pytest.approx(0.0)
    assert tr.net_points == pytest.approx(-0.05)  # 2 * (0.02 + 0.005)


# --- signal semantics ----------------------------------------------------

def test_lower_band_touch_arms_a_long_not_a_short():
    """A flush to the lower band is a LONG setup once the EMA reclaims VWAP."""
    prices = np.concatenate([
        np.linspace(150, 148, 12),   # sell off toward the lower band
        np.linspace(148, 151, 20),   # reclaim
    ])
    df = make_session(prices, volumes=[2000] * len(prices))
    feat = build_features(df)
    sigs = generate_signals(feat, Config(arm_bars=30))
    assert any(s.direction == 1 for s in sigs), "expected a long from the lower-band fade"


def test_arm_expires_after_arm_bars():
    rng = np.random.default_rng(3)
    prices = 150 + np.cumsum(rng.normal(0, 0.3, 80))
    df = make_session(prices, volumes=rng.integers(500, 5000, 80))
    feat = build_features(df)

    short_arm = generate_signals(feat, Config(arm_bars=2))
    long_arm = generate_signals(feat, Config(arm_bars=40))
    assert len(short_arm) <= len(long_arm)


def test_direction_flags_are_respected():
    rng = np.random.default_rng(11)
    prices = 150 + np.cumsum(rng.normal(0, 0.4, 100))
    df = make_session(prices, volumes=rng.integers(500, 5000, 100))
    feat = build_features(df)

    longs = generate_signals(feat, Config(allow_short=False))
    shorts = generate_signals(feat, Config(allow_long=False))
    assert all(s.direction == 1 for s in longs)
    assert all(s.direction == -1 for s in shorts)


# --- metrics -------------------------------------------------------------

def test_metrics_match_hand_computed_values():
    trades = pd.DataFrame({
        "net_points": [2.0, -1.0, 3.0, -2.0],
        "entry_price": [100.0] * 4,
        "stop": [99.0] * 4,
        "direction": [1, 1, -1, -1],
    })
    m = metrics(trades, n_sessions=10)
    assert m["trades"] == 4
    assert m["win_rate"] == 50.0
    assert m["profit_factor"] == pytest.approx(5.0 / 3.0, abs=1e-3)  # 5 won / 3 lost
    assert m["total_points"] == pytest.approx(2.0)
    assert m["points_per_contract"] == pytest.approx(200.0)  # x100 shares
    # Equity path 2, 1, 4, 2 -> peak 4, trough after = 2 -> drawdown 2.
    assert m["max_drawdown_points"] == pytest.approx(2.0)


def test_empty_trades_produce_zeroed_metrics_not_a_crash():
    m = metrics(pd.DataFrame(), n_sessions=120)
    assert m["trades"] == 0
    assert m["profit_factor"] == 0.0
    assert m["total_points"] == 0.0


def test_points_per_contract_is_exactly_100x_points():
    rng = np.random.default_rng(5)
    trades = pd.DataFrame({
        "net_points": rng.normal(0, 1, 50),
        "entry_price": [150.0] * 50,
        "stop": [149.0] * 50,
        "direction": rng.choice([1, -1], 50),
    })
    m = metrics(trades, n_sessions=120)
    assert m["points_per_contract"] == pytest.approx(m["total_points"] * 100, abs=0.01)


# --- integration ---------------------------------------------------------

def test_full_run_on_multi_session_random_walk():
    rng = np.random.default_rng(99)
    frames = []
    for i in range(20):
        day = pd.Timestamp("2024-03-04") + pd.Timedelta(days=i)
        if day.weekday() >= 5:
            continue
        prices = 150 + np.cumsum(rng.normal(0, 0.25, 78))
        frames.append(make_session(prices, volumes=rng.integers(500, 5000, 78),
                                   day=str(day.date())))
    df = pd.concat(frames)

    cfg = Config()
    sessions = prepare_sessions(df, cfg)
    trades = run(sessions, cfg)
    m = metrics(trades, len(sessions))

    assert m["trades"] == len(trades)
    if not trades.empty:
        # Every trade must exit at or after entry, and stay inside its session.
        assert (trades["exit_time"] >= trades["entry_time"]).all()
        assert (trades["exit_time"].dt.normalize() == trades["session"]).all()
