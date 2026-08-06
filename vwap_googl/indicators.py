"""Session-anchored VWAP bands and the supporting indicators the strategy filters on.

Everything here operates on an intraday bar frame indexed by a tz-aware
DatetimeIndex in US/Eastern, with columns: open, high, low, close, volume.
All session-scoped calculations reset at the RTH open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RTH_START = "09:30"
RTH_END = "16:00"


def session_key(index: pd.DatetimeIndex) -> pd.Series:
    """Trading-date label used to reset every session-anchored calculation."""
    return pd.Series(index.normalize(), index=index)


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def vwap_bands(df: pd.DataFrame, devs=(1.0, 2.0, 3.0)) -> pd.DataFrame:
    """Session VWAP plus volume-weighted standard deviation bands.

    The dispersion is the volume-weighted RMS of typical price around the
    running VWAP -- the same definition charting platforms use for their
    "VWAP standard deviation bands", not a rolling close-to-close stdev.
    """
    tp = typical_price(df)
    vol = df["volume"].astype("float64")
    grp = session_key(df.index)

    cum_vol = vol.groupby(grp).cumsum()
    cum_pv = (tp * vol).groupby(grp).cumsum()
    vwap = cum_pv / cum_vol.replace(0.0, np.nan)

    # E[p^2] - (E[p])^2 under the volume measure.
    cum_pv2 = (tp.pow(2) * vol).groupby(grp).cumsum()
    variance = (cum_pv2 / cum_vol.replace(0.0, np.nan)) - vwap.pow(2)
    sigma = np.sqrt(variance.clip(lower=0.0))

    out = pd.DataFrame({"vwap": vwap, "vwap_sigma": sigma}, index=df.index)
    for d in devs:
        tag = str(d).rstrip("0").rstrip(".")
        out[f"vwap_up{tag}"] = vwap + d * sigma
        out[f"vwap_dn{tag}"] = vwap - d * sigma
    return out


def ema(series: pd.Series, span: int) -> pd.Series:
    """Session-reset EMA. Carrying an EMA across the close would let the prior
    day's level dictate the first crosses of the new session."""
    grp = session_key(series.index)
    return series.groupby(grp, group_keys=False).apply(
        lambda s: s.ewm(span=span, adjust=False).mean()
    )


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ADX -- used to separate ranging sessions (fade works) from
    trending sessions (fading the band is how you get run over)."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = atr(df, period)
    alpha = 1.0 / period
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / tr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / tr

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False).mean().fillna(0.0)


def relative_volume(df: pd.DataFrame, lookback_sessions: int = 20) -> pd.Series:
    """Bar volume vs. the average volume for that same time-of-day slot over
    the trailing N sessions. Flags genuine participation rather than the
    mechanical volume smile of the open and close."""
    grp = session_key(df.index)
    slot = pd.Series(df.index.strftime("%H:%M"), index=df.index)
    frame = pd.DataFrame({"vol": df["volume"].astype("float64"), "slot": slot, "day": grp})

    baseline = (
        frame.groupby("slot")["vol"]
        .transform(lambda s: s.shift(1).rolling(lookback_sessions, min_periods=5).mean())
    )
    return (frame["vol"] / baseline).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def vwap_slope(vwap: pd.Series, bars: int = 6) -> pd.Series:
    """VWAP drift over the last N bars, normalised by price so it is
    comparable across sessions. Near zero => balanced/rotational session."""
    grp = session_key(vwap.index)
    delta = vwap.groupby(grp).diff(bars)
    return (delta / vwap).fillna(0.0)


def minutes_from_open(index: pd.DatetimeIndex) -> pd.Series:
    open_ts = index.normalize() + pd.Timedelta(hours=9, minutes=30)
    return pd.Series((index - open_ts).total_seconds() / 60.0, index=index)


def build_features(df: pd.DataFrame, ema_span: int = 9) -> pd.DataFrame:
    """Attach every indicator the strategy and its filters may reference."""
    feat = df.copy()
    feat = feat.join(vwap_bands(feat))
    feat["ema9"] = ema(feat["close"], ema_span)
    feat["atr"] = atr(feat)
    feat["rsi"] = rsi(feat["close"])
    feat["adx"] = adx(feat)
    feat["rvol"] = relative_volume(feat)
    feat["vwap_slope"] = vwap_slope(feat["vwap"])
    feat["band_width"] = (feat["vwap_up2"] - feat["vwap_dn2"]) / feat["vwap"]
    feat["min_from_open"] = minutes_from_open(feat.index)
    feat["ema_vs_vwap"] = feat["ema9"] - feat["vwap"]
    feat["session"] = session_key(feat.index).values
    return feat
