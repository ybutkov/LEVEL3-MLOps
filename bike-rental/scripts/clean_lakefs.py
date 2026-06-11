"""Remove leftover prototype data from LakeFS ``main`` (run once).

The week-4 notebook prototype uploaded model splits under ``datasets/`` on
``main``; the pipeline now manages only ``raw/`` (seeded via
``scripts/seed_lakefs.py``). This deletes the stale prefix so the repo holds only
pipeline-managed data. Targeted delete on ``main`` — the repo and its history are
kept (no repo recreate, which would leave storage-namespace objects behind).

Credentials come from ``LAKEFS_ACCESS_KEY`` / ``LAKEFS_SECRET_KEY`` (``.env``
auto-loaded, gitignored). Usage::

    uv run python scripts/clean_lakefs.py
"""

import os

import lakefs
from dotenv import load_dotenv
from lakefs.client import Client

from bike_rental.config import AppConfig

# Stale prefix left by the notebook prototype; the pipeline only manages `raw/`.
PRUNE_PREFIX = "datasets/"


def main() -> None:
    load_dotenv()  # read LAKEFS_* from .env (same file Dagster auto-loads)
    cfg = AppConfig.load()
    client = Client(
        host=cfg.lakefs.host,
        username=os.environ["LAKEFS_ACCESS_KEY"],
        password=os.environ["LAKEFS_SECRET_KEY"],
    )
    branch = lakefs.Repository(cfg.lakefs.repo, client=client).branch(cfg.lakefs.merge_into)

    paths = [obj.path for obj in branch.objects(prefix=PRUNE_PREFIX)]
    if not paths:
        print(f"nothing to clean under {PRUNE_PREFIX!r} on {cfg.lakefs.merge_into}")
        return

    branch.delete_objects(paths)
    commit = branch.commit(message=f"clean prototype data under {PRUNE_PREFIX}")
    print(f"removed {len(paths)} object(s) under {PRUNE_PREFIX!r}; commit {commit.id}")


if __name__ == "__main__":
    main()
