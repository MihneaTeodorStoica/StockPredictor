from __future__ import annotations
from pathlib import Path
import pandas as pd
import logging

log = logging.getLogger(__name__)

def load_data(csv_paths: list[str | Path]) -> list[pd.DataFrame]:
    """
    Loads each path in *csv_paths* into a pandas DataFrame.

    Rows with *all* NaNs are dropped immediately to save RAM.
    """
    dataframes: list[pd.DataFrame] = []

    for path in csv_paths:
        path = Path(path)
        try:
            df = pd.read_csv(path).dropna(how="all")
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="ignore")
            dataframes.append(df)
        except Exception as exc:
            log.warning("Failed to load %s: %s", path, exc)

    log.info("Loaded %d/%d CSVs", len(dataframes), len(csv_paths))
    return dataframes
