from __future__ import annotations
from pathlib import Path
import zipfile
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

def extract_archive(
    zip_path: str | Path = "data/archive.zip",
    extract_dir: str | Path = "data/extracted",
    exclude_prefix: str = "Data/",
) -> None:
    """
    Extracts *zip_path* into *extract_dir*, skipping any members that start with
    *exclude_prefix*.  After extraction, every ``*.txt`` file is renamed to ``*.csv``.

    This runs every time – idempotent because zipfile will overwrite duplicates.
    """
    zip_path   = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # ── Extraction ────────────────────────────────────────────────────────────
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.startswith(exclude_prefix)]
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, path=extract_dir)

    # ── Rename *.txt → *.csv ──────────────────────────────────────────────────
    for txt in extract_dir.rglob("*.txt"):
        csv_path = txt.with_suffix(".csv")
        txt.rename(csv_path)
        log.debug("Renamed %s → %s", txt.name, csv_path.name)

    log.info("Archive extracted to %s (txt→csv rename complete)", extract_dir)
