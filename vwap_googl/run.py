"""End-to-end study: build features, sweep variants in-sample, validate the
winner out-of-sample, and print the scorecard.

Usage:
    python -m vwap_googl.run --data path/to/GOOGL_5min.csv
    python -m vwap_googl.run --data "data/*.csv" --interval 5min
    python -m vwap_googl.run --data data/ --train 120 --test 120

The split is strictly chronological: the first `--train` sessions select the
variant, the next `--test` sessions are touched exactly once, to report. No
variant is ever chosen using out-of-sample information.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import data as data_mod
from .backtest import metrics, prepare_sessions, run
from .variants import combo_variants, core_variants


def min_trades_for(n_sessions: int) -> int:
    """Trade-count floor below which in-sample stats are noise.

    Scaled to the study length: a filter that fires four times over 120
    sessions can post a 100% win rate and mean nothing.
    """
    return max(20, n_sessions // 4)


def composite_score(m: dict, min_trades: int) -> float:
    """Single ranking number balancing the five things asked for.

    Profit factor and expectancy carry the most weight because win rate alone
    is trivially gamed by a wide stop and a tiny target. Drawdown enters as a
    penalty on total return.
    """
    if m["trades"] < min_trades:
        return -1e9

    pf = min(m["profit_factor"], 5.0) if np.isfinite(m["profit_factor"]) else 5.0
    dd = max(m["max_drawdown_points"], 1e-6)
    recovery = m["total_points"] / dd  # return per unit of pain

    return (
        1.5 * pf
        + 1.0 * np.clip(recovery, -5.0, 5.0)
        + 2.0 * np.clip(m["expectancy_r"], -1.0, 1.0)
        + 0.02 * m["win_rate"]
    )


def split_sessions(sessions: list, train: int, test: int) -> tuple[list, list]:
    if len(sessions) < train + test:
        raise ValueError(
            f"Need {train + test} sessions, have {len(sessions)}. "
            f"Reduce --train/--test or supply more data."
        )
    ordered = sorted(sessions)
    return ordered[:train], ordered[train : train + test]


def evaluate(feat_by_session: dict, days: list, cfg) -> tuple[dict, pd.DataFrame]:
    subset = {d: feat_by_session[d] for d in days if d in feat_by_session}
    trades = run(subset, cfg)
    return metrics(trades, len(subset)), trades


def format_table(rows: list[dict], cols: list[str]) -> str:
    widths = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in cols}
    head = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join(
        "  ".join(f"{r.get(c, '')}".ljust(widths[c]) for c in cols) for r in rows
    )
    return f"{head}\n{sep}\n{body}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="GOOGL VWAP 2-sigma / 9-EMA study")
    p.add_argument("--data", required=True, help="CSV file, directory or glob")
    p.add_argument("--interval", default="5min", help="Bar size, e.g. 1min/5min/15min")
    p.add_argument("--train", type=int, default=120, help="In-sample sessions")
    p.add_argument("--test", type=int, default=120, help="Out-of-sample sessions")
    p.add_argument("--tz", default="US/Eastern", help="Timezone of naive timestamps")
    p.add_argument("--out", default="results", help="Output directory")
    p.add_argument("--top", type=int, default=12, help="Variants to show")
    p.add_argument("--force", action="store_true", help="Run despite data-quality warnings")
    args = p.parse_args(argv)

    print(f"Loading {args.data} ...")
    raw = data_mod.load_any(args.data, assume_tz=args.tz)
    bars = data_mod.regular_hours(raw)
    if args.interval:
        bars = data_mod.resample(bars, args.interval)
        bars = data_mod.regular_hours(bars)

    info = data_mod.audit(bars)
    print("\n=== DATA AUDIT ===")
    for k, v in info.items():
        print(f"  {k:28s} {v}")

    problems = data_mod.validate_for_backtest(bars, args.train + args.test)
    if problems:
        print("\n=== DATA PROBLEMS ===")
        for prob in problems:
            print(f"  ! {prob}")
        if not args.force:
            print("\nAborting. Re-run with --force to proceed anyway (results will be unreliable).")
            return 1

    from .variants import BASE

    feat_by_session = prepare_sessions(bars, BASE)
    sessions = list(feat_by_session.keys())
    train_days, test_days = split_sessions(sessions, args.train, args.test)

    min_trades = min_trades_for(len(train_days))
    print(f"\nIn-sample:      {train_days[0].date()} -> {train_days[-1].date()} ({len(train_days)} sessions)")
    print(f"Out-of-sample:  {test_days[0].date()} -> {test_days[-1].date()} ({len(test_days)} sessions)")
    print(f"Minimum trades for a variant to qualify: {min_trades}")

    # --- stage 1: single-factor sweep, in-sample only --------------------
    print("\n=== STAGE 1: single-factor sweep (in-sample) ===")
    results = []
    for cfg in core_variants():
        m, _ = evaluate(feat_by_session, train_days, cfg)
        m["name"] = cfg.name
        m["score"] = round(composite_score(m, min_trades), 3)
        results.append((cfg, m))

    ranked = sorted(results, key=lambda x: x[1]["score"], reverse=True)
    cols = ["name", "trades", "win_rate", "profit_factor", "total_points",
            "points_per_contract", "max_drawdown_points", "expectancy_r", "score"]
    print(format_table([m for _, m in ranked[: args.top]], cols))

    # --- stage 2: combinations -------------------------------------------
    print("\n=== STAGE 2: stacked combinations (in-sample) ===")
    winners = [m["name"] for _, m in ranked[:8]]
    combo_results = []
    for cfg in combo_variants(winners):
        m, _ = evaluate(feat_by_session, train_days, cfg)
        m["name"] = cfg.name
        m["score"] = round(composite_score(m, min_trades), 3)
        combo_results.append((cfg, m))

    all_ranked = sorted(results + combo_results, key=lambda x: x[1]["score"], reverse=True)
    print(format_table([m for _, m in sorted(combo_results, key=lambda x: x[1]["score"], reverse=True)], cols))

    # --- stage 3: out-of-sample validation of the single winner ----------
    best_cfg, best_is = all_ranked[0]
    print(f"\n=== STAGE 3: out-of-sample validation ===")
    print(f"Selected on in-sample only: '{best_cfg.name}'")

    best_oos, oos_trades = evaluate(feat_by_session, test_days, best_cfg)
    base_is, _ = evaluate(feat_by_session, train_days, core_variants()[0])
    base_oos, _ = evaluate(feat_by_session, test_days, core_variants()[0])

    comparison = [
        {**base_is, "name": "base (IS)"},
        {**base_oos, "name": "base (OOS)"},
        {**best_is, "name": f"{best_cfg.name} (IS)"},
        {**best_oos, "name": f"{best_cfg.name} (OOS)"},
    ]
    print()
    print(format_table(comparison, cols[:-1]))

    print(f"\nExpectancy (R/trade):  IS {best_is['expectancy_r']:+.3f}  ->  OOS {best_oos['expectancy_r']:+.3f}")
    print(f"Profit factor:         IS {best_is['profit_factor']:.3f}  ->  OOS {best_oos['profit_factor']:.3f}")

    if best_oos["trades"] < min_trades_for(len(test_days)):
        print("VERDICT: too few out-of-sample trades to judge. Not tradeable evidence.")
    elif best_oos["profit_factor"] < 1.0:
        print("VERDICT: the selected variant did NOT hold up out-of-sample. Treat as overfit.")
    elif best_is["expectancy_r"] > 0 and best_oos["expectancy_r"] < 0.5 * best_is["expectancy_r"]:
        print("VERDICT: edge more than halved out-of-sample. Fragile; size down or re-test.")
    elif best_oos["expectancy_r"] > best_is["expectancy_r"]:
        print("VERDICT: held up out-of-sample (OOS better than IS -- likely regime luck, not skill).")
    else:
        print("VERDICT: edge persisted out-of-sample.")

    # --- persist ----------------------------------------------------------
    import os

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame([m for _, m in all_ranked]).to_csv(f"{args.out}/in_sample_ranking.csv", index=False)
    oos_trades.to_csv(f"{args.out}/oos_trades.csv", index=False)
    with open(f"{args.out}/best_config.json", "w") as fh:
        json.dump({"config": best_cfg.to_dict(), "in_sample": best_is, "out_of_sample": best_oos}, fh, indent=2, default=str)
    print(f"\nWrote {args.out}/in_sample_ranking.csv, oos_trades.csv, best_config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
