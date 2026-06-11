"""Seed raw CSVs into LakeFS through a validated ingest branch (run once).

Demonstrates the branching strategy that protects the trunk (``merge_into``): new
raw data lands on the ``ingest`` branch, is validated against the same Pandera
schemas the pipeline uses, and is merged into the trunk only if every file passes
— so bad data never reaches the trunk. Re-run whenever the raw files change.

Credentials come from ``LAKEFS_ACCESS_KEY`` / ``LAKEFS_SECRET_KEY`` — put them in a
``.env`` (auto-loaded, gitignored) or export them. Usage::

    uv run python scripts/seed_lakefs.py
"""

import io
import os
from pathlib import Path

import lakefs
import pandas as pd
from dotenv import load_dotenv
from lakefs.client import Client

from bike_rental.config import AppConfig
from bike_rental.defs.schemas import HolidaysRaw, RentalsRaw, WeatherRaw

# Raw file -> the Pandera schema it must satisfy (same schemas as the asset checks).
FILES = {
    "registered_bike_rentals.csv": RentalsRaw,
    "direct_pickup_bike_rentals.csv": RentalsRaw,
    "weather.csv": WeatherRaw,
    "holidays.csv": HolidaysRaw,
}


def main() -> None:
    """Seed local raw CSVs into LakeFS through the validated ingest branch."""
    load_dotenv()
    cfg = AppConfig.load()
    client = Client(
        host=cfg.lakefs.host,
        username=os.environ["LAKEFS_ACCESS_KEY"],
        password=os.environ["LAKEFS_SECRET_KEY"],
    )
    repo = lakefs.Repository(cfg.lakefs.repo, client=client)
    src_dir = Path(cfg.source_dir)
    prefix = cfg.lakefs.raw_prefix
    ingest = cfg.lakefs.ingest_branch

    # 1. ingest branch off the trunk (quarantine zone for incoming new data)
    branch = repo.branch(ingest).create(source_reference=cfg.lakefs.merge_into, exist_ok=True)

    # 2. upload local raw onto the {ingest} branch and commit
    for filename in FILES:
        branch.object(f"{prefix}/{filename}").upload(
            (src_dir / filename).read_bytes(), pre_sign=False
        )
    commit = branch.commit(message="ingest raw batch")
    print(f"ingest commit: {commit.id}")

    # 3. validate what landed on ingest; a failure -> no merge -> trunk - clean
    for filename, schema in FILES.items():
        data = branch.object(f"{prefix}/{filename}").reader(pre_sign=False).read()
        df = pd.read_csv(io.BytesIO(data))
        schema.validate(df, lazy=True)
        print(f"validated {filename}: {len(df)} rows")

    # 4. all passed -> merge ingest into the trunk
    merge_commit = repo.branch(ingest).merge_into(cfg.lakefs.merge_into)
    print(f"merged {ingest} -> {cfg.lakefs.merge_into}: {merge_commit}")


if __name__ == "__main__":
    main()
