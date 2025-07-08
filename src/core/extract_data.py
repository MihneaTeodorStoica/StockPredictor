import zipfile
from tqdm import tqdm
import os

def extract_archive():
    zip_path = "data/archive.zip"
    extract_path = "data/extracted"
    exclude_prefix = "Data/"

    # ── Extracting Files ─────────────────────────────────────
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = [
            member for member in zip_ref.namelist()
            if not member.startswith(exclude_prefix)
        ]

        for member in tqdm(members, desc="Extracting", unit="file"):
            zip_ref.extract(member, extract_path)

    # ── Renaming .txt to .csv ───────────────────────────────
    for dirpath, _, filenames in os.walk(extract_path):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, filename[:-4] + ".csv")
                os.rename(old_path, new_path)
