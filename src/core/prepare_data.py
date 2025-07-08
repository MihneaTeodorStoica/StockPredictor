from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import logging

log = logging.getLogger(__name__)

def prepare_training_data(
    dfs: list[pd.DataFrame],
    *,
    window_size: int = 10,
    save_dir: str | Path = "data/log",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds an (X, y) dataset where each X‐row contains *window_size* consecutive
    percentage changes of the *Close* price, and *y* is the subsequent change.

    The resulting arrays are saved as ``X.npy`` & ``y.npy`` under *save_dir*.

    Returns
    -------
    X : np.ndarray  shape = (samples, window_size)
    y : np.ndarray  shape = (samples,)
    """
    X, y = [], []

    for df in dfs:
        if "Close" not in df.columns:
            continue

        prices = df["Close"].dropna().to_numpy(float)
        if prices.size < window_size + 1:
            continue

        pct = np.diff(prices) / prices[:-1]        # daily % change
        if pct.size < window_size + 1:
            continue

        for i in range(pct.size - window_size):
            X.append(pct[i : i + window_size])
            y.append(pct[i + window_size])

    if not X:
        log.warning("No valid sequences produced – check your data / window_size.")
        return np.empty((0, window_size)), np.empty((0,))

    X = np.stack(X)
    y = np.array(y)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "X.npy", X)
    np.save(save_dir / "y.npy", y)

    log.info("Prepared dataset: X%s  y%s  (saved to %s)", X.shape, y.shape, save_dir)
    return X, y
