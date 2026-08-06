"""Loading and normalising GOOGL intraday bars.

The loader is deliberately format-tolerant: TWS/IB exports, Yahoo dumps and
generic OHLCV CSVs all land on the same normalised frame so the rest of the
package never has to care where the bars came from.

Normalised contract:
    tz-aware DatetimeIndex in US/Eastern, sorted, unique
    columns: open, high, low, close, volume
    regular trading hours only (09:30 - 16:00 ET)
"""

from __future__ import annotations

import glob
import os

import pandas as pd

EASTERN = "US/Eastern"

# Every spelling of the same six fields I have run into across TWS, IB Gateway
# exports, Yahoo, Polygon and the usual Kaggle dumps.
_COLUMN_ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "date_time": "timestamp",
    "time": "timestamp",
    "timestamp": "timestamp",
    "bar_time": "timestamp",
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "last": "close",
    "adj close": "close",
    "volume": "volume",
    "v": "volume",
    "vol": "volume",
}

_REQUIRED = ["open", "high", "low", "close", "volume"]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("<", "").replace(">", "")
        if key in _COLUMN_ALIASES:
            target = _COLUMN_ALIASES[key]
            # Never let "adj close" clobber a real close column.
            if target == "close" and key == "adj close" and "close" in renamed.values():
                continue
            renamed[col] = target
    out = df.rename(columns=renamed)
    return out.loc[:, ~out.columns.duplicated()]


def _build_timestamp(df: pd.DataFrame) -> pd.Series:
    """Handles both a single datetime column and split date/time columns."""
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")

    lower = {str(c).strip().lower(): c for c in df.columns}
    if "date" in lower and "time" in lower:
        combined = (
            df[lower["date"]].astype(str).str.strip()
            + " "
            + df[lower["time"]].astype(str).str.strip()
        )
        return pd.to_datetime(combined, errors="coerce", format="mixed")

    raise ValueError(
        f"No timestamp column found. Columns present: {list(df.columns)}"
    )


def load_csv(path: str, assume_tz: str = EASTERN) -> pd.DataFrame:
    """Read one CSV of intraday bars into the normalised frame."""
    raw = pd.read_csv(path)
    df = _normalise_columns(raw)

    ts = _build_timestamp(df)
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"])

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    df = df[["timestamp"] + _REQUIRED].copy()
    for col in _REQUIRED:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    df = df.set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize(assume_tz, ambiguous="NaT", nonexistent="NaT")
    else:
        df.index = df.index.tz_convert(assume_tz)

    df = df[df.index.notna()]
    return df[~df.index.duplicated(keep="first")]


def load_dir(pattern: str, assume_tz: str = EASTERN) -> pd.DataFrame:
    """Load and concatenate every CSV matching a glob (e.g. 'data/GOOGL*.csv')."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {pattern}")
    frames = [load_csv(p, assume_tz) for p in paths]
    combined = pd.concat(frames).sort_index()
    return combined[~combined.index.duplicated(keep="first")]


def load_any(path: str, assume_tz: str = EASTERN) -> pd.DataFrame:
    """Accept a file, a directory or a glob."""
    if os.path.isdir(path):
        return load_dir(os.path.join(path, "*.csv"), assume_tz)
    if any(ch in path for ch in "*?["):
        return load_dir(path, assume_tz)
    return load_csv(path, assume_tz)


def regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pre/post market. Session VWAP is only meaningful anchored to the
    RTH open, and extended-hours bars are thin enough to distort the bands."""
    return df.between_time("09:30", "15:59")


def resample(df: pd.DataFrame, interval: str = "5min") -> pd.DataFrame:
    """Aggregate to the target bar size, aligned to the 09:30 open."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = (
        df.groupby(df.index.normalize(), group_keys=False)
        .apply(lambda day: day.resample(interval, origin=day.index[0]).agg(agg))
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def audit(df: pd.DataFrame) -> dict:
    """Data-quality report. A VWAP strategy is only as good as its volume
    field, so zero-volume bars are called out explicitly."""
    sessions = df.index.normalize()
    per_session = sessions.value_counts()
    zero_vol = int((df["volume"] <= 0).sum())
    return {
        "rows": len(df),
        "sessions": int(sessions.nunique()),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "zero_volume_bars": zero_vol,
        "zero_volume_pct": round(100.0 * zero_vol / max(len(df), 1), 2),
        "bars_per_session_median": int(per_session.median()) if len(per_session) else 0,
        "bars_per_session_min": int(per_session.min()) if len(per_session) else 0,
        "bars_per_session_max": int(per_session.max()) if len(per_session) else 0,
    }


def validate_for_backtest(df: pd.DataFrame, required_sessions: int = 240) -> list[str]:
    """Return a list of blocking problems; empty list means good to go."""
    problems = []
    info = audit(df)

    if info["sessions"] < required_sessions:
        problems.append(
            f"Only {info['sessions']} sessions available, need {required_sessions} "
            f"(120 in-sample + 120 out-of-sample)."
        )
    if info["zero_volume_pct"] > 5.0:
        problems.append(
            f"{info['zero_volume_pct']}% of bars have zero volume. VWAP is "
            f"volume-weighted, so these bars contribute nothing and the bands "
            f"will be wrong."
        )
    if info["bars_per_session_median"] < 20:
        problems.append(
            f"Median {info['bars_per_session_median']} bars/session is too sparse "
            f"for intraday signals."
        )
    return problems
