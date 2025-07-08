from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_close_prices(
    dfs: list[pd.DataFrame],
    *,
    labels: list[str] | None = None,
    show_legend: bool = False,
) -> None:
    """
    Plots the ``Close`` column from every DataFrame in *dfs*.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        The dataframes to plot (must contain a ``Close`` column).
    labels : list[str] | None
        Optional custom series labels.  Defaults to “Series 1”, “Series 2”, …
    show_legend : bool
        Whether to display the legend.  Kept False by default per user request.
    """
    if labels and len(labels) != len(dfs):
        raise ValueError("labels length must match dfs length")

    plt.figure(figsize=(12, 6))

    for i, df in enumerate(dfs):
        label = labels[i] if labels else f"Series {i+1}"
        if "Date" in df.columns:
            plt.plot(df["Date"], df["Close"], label=label, linewidth=0.8)
        else:
            plt.plot(df["Close"].reset_index(drop=True), label=label, linewidth=0.8)

    plt.title("Close-Price Comparison")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    if show_legend:
        plt.legend(loc="upper left", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()
