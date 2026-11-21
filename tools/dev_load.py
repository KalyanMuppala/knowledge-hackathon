"""Load the snapshots through the harness loader (shared by the other dev scripts)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

from run_eval import load_snapshots  # noqa: E402


def load(snap_dir=ROOT / "snapshots"):
    return load_snapshots(Path(snap_dir))


def build_system(use_llm: bool = False):
    import solution
    ks = solution.build()
    if not use_llm:
        ks.llm.client = None
    for snap in load():
        ks.ingest(snap)
    return ks
