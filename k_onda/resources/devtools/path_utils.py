# k_onda/devtools/path_utils.py
from pathlib import Path

def find_project_root(marker="README.md"):
    here = Path.cwd()
    for p in [here] + list(here.parents):
        if (p / marker).exists():
            return p
    raise RuntimeError(f"Could not find project root using marker: {marker}")
