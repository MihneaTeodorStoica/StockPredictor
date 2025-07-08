from __future__ import annotations
from pathlib import Path
from glob import glob
import random
import logging

log = logging.getLogger(__name__)

_ETF_GLOB   = "data/extracted/ETFs/**/*.csv"
_STOCK_GLOB = "data/extracted/Stocks/**/*.csv"

def select_random_csvs(n: int = 100) -> list[str]:
    """
    Returns up to *n* random CSV paths drawn from *ETFs* and *Stocks*
    sub-directories under ``data/extracted``.
    """
    csv_files = glob(_ETF_GLOB, recursive=True) + glob(_STOCK_GLOB, recursive=True)
    if not csv_files:
        log.warning("No CSVs found under data/extracted – did you run extract_archive?")
        return []

    chosen = random.sample(csv_files, k=min(n, len(csv_files)))
    log.info("Randomly selected %d CSV files", len(chosen))
    return chosen
