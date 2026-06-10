"""Git helpers: capture the code revision logged alongside each training run."""

import subprocess


def get_git_commit() -> str:
    """Return the current commit hash, suffixed ``-dirty`` if the tree is modified.

    Returns
    -------
    str
        ``<hash>`` for a clean tree, ``<hash>-dirty`` if there are uncommitted
        changes, or ``"unknown"`` if git is unavailable (e.g. not a repo).
    """
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        return f"{commit}-dirty" if status else commit
    except Exception:
        return "unknown"
