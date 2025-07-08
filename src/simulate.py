#!/usr/bin/env python3
"""
Batch trading simulation with the trained model.

• Loads many CSVs (randomly via core.analyze_data.select_random_csvs, or paths supplied on CLI).
• Runs the same buy / sell / hold logic.
• Aggregates results and produces summary graphics in data/log/.

Safe: no shorting, only 0 % or 100 % long.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from core.analyze_data import select_random_csvs

# ───────────── Config ─────────────
WINDOW_SIZE = 10           # must match training
THRESH      = 0.002        # 0.2 % swing to trade
START_CASH  = 1.0
MODEL_PATH  = Path("data/model/simple_model.keras")
LOG_DIR     = Path("data/log")
# ──────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("simulate")


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation on CLOSE-price array
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_series(
    closes: np.ndarray,
    model,
    window_size: int = WINDOW_SIZE,
    thresh: float = THRESH,
    start_cash: float = START_CASH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (equity_curve, bh_curve)."""
    pct = np.diff(closes) / closes[:-1]
    if len(pct) < window_size + 1:
        raise ValueError("Series too short.")

    cash, shares = start_cash, 0.0
    equity_curve: List[float] = []

    for i in range(window_size, len(pct)):
        window = pct[i - window_size : i]
        pred   = model.predict(window[None, :], verbose=0)[0, 0]
        price  = closes[i]

        # buy / sell / hold decision
        if pred >  thresh and shares == 0:
            shares = cash / price
            cash   = 0.0
        elif pred < -thresh and shares > 0:
            cash   = shares * price
            shares = 0.0
        # else HOLD

        equity_curve.append(cash + shares * price)

    # buy-and-hold baseline (from first trading day)
    bh_shares = start_cash / closes[window_size]
    bh_curve  = bh_shares * closes[window_size : window_size + len(equity_curve)]

    return np.asarray(equity_curve), np.asarray(bh_curve)


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────
def run_batch(csv_paths: Iterable[Path]) -> None:
    """Simulate on many CSVs and write graphics + summary."""
    csv_paths = list(csv_paths)
    if not csv_paths:
        log.error("No CSV files provided.")
        return

    model = load_model(MODEL_PATH)
    ai_curves, bh_curves, ratios = [], [], []   # type: List[np.ndarray]

    for path in csv_paths:
        try:
            closes = (
                pd.read_csv(path)["Close"]
                .dropna()
                .values
                .astype(np.float64)
            )
            eq_ai, eq_bh = _simulate_series(closes, model)
            ai_curves.append(eq_ai)
            bh_curves.append(eq_bh)
            ratios.append(eq_ai[-1] / eq_bh[-1])
            log.info("%s  |  %.2f → %.2f (AI) vs %.2f (B&H)",
                     path.name, START_CASH, eq_ai[-1], eq_bh[-1])
        except Exception as exc:
            log.warning("Skipping %s (%s)", path.name, exc)

    if not ai_curves:
        log.error("No successful simulations.")
        return

    # Align curves by truncating to shortest length
    min_len = min(map(len, ai_curves))
    ai_mat  = np.vstack([c[:min_len] for c in ai_curves])
    bh_mat  = np.vstack([c[:min_len] for c in bh_curves])
    mean_ai = ai_mat.mean(axis=0)
    mean_bh = bh_mat.mean(axis=0)

    # ── Plot: average equity curves ──────────────────────────
    plt.figure(figsize=(12, 6))
    plt.plot(mean_ai, label="Mean AI strategy")
    plt.plot(mean_bh, label="Mean Buy & Hold")
    plt.title("Mean Equity Curve over %d tickers" % len(ai_curves))
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOG_DIR / "simulation_vs_bh_mean.png")

    # ── Plot: histogram of outperformance ───────────────────
    plt.figure(figsize=(8, 5))
    plt.hist((np.array(ratios) - 1) * 100, bins=30, edgecolor="black")
    plt.xlabel("AI Outperformance over B&H (%)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Strategy Outperformance")
    plt.tight_layout()
    plt.savefig(LOG_DIR / "outperformance_hist.png")

    # ── Plot: individual equity curves (faint) ──────────────
    plt.figure(figsize=(12, 6))
    for c in ai_curves:
        plt.plot(c, alpha=0.3)
    plt.title("Individual AI Equity Curves")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOG_DIR / "individual_curves.png")
    plt.close("all")

    # ── Summary stats ───────────────────────────────────────
    ratios = np.array(ratios)
    win_rate = (ratios > 1).mean() * 100
    log.info("\n=== Aggregate Results (%d tickers) ===", len(ai_curves))
    log.info("Average final value  – AI : $%.4f", mean_ai[-1])
    log.info("Average final value  – B&H: $%.4f", mean_bh[-1])
    log.info("Mean outperformance  : %+0.2f %%", (ratios.mean() - 1) * 100)
    log.info("Median outperformance: %+0.2f %%", (np.median(ratios) - 1) * 100)
    log.info("Win-rate (AI > B&H)  : %.1f %%", win_rate)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch trading simulation.")
    p.add_argument(
        "paths", nargs="*", type=Path,
        help="CSV files or directories containing CSVs."
    )
    p.add_argument(
        "--random", "-r", type=int, metavar="N",
        help="Ignore PATHS and select N random CSVs via core.analyze_data."
    )
    return p.parse_args()


def main() -> None:
    args = parse_cli()

    if args.random:
        csv_paths = map(Path, select_random_csvs(args.random))
    else:
        # Gather files from provided paths; expand directories
        files: List[Path] = []
        for p in args.paths:
            if p.is_dir():
                files.extend(p.rglob("*.csv"))
            else:
                files.append(p)
        csv_paths = files if files else map(Path, select_random_csvs(1))

    run_batch(csv_paths)


if __name__ == "__main__":
    main()
