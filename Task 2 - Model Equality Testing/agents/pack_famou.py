"""Assemble and validate a development-only Famou project. Never contact Baidu."""

from pathlib import Path
import shutil
import sys

from famou_sdk.cli.utils import ProjectManager


def pack(out):
    root = Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=False)
    for name in ("init.py", "evaluator.py", "problem.md", "prompt.md", "config.yaml"):
        shutil.copy(root / "famou" / name, out / name)
    for name in ("arena.py", "development.json"):
        shutil.copy(root / name, out / name)
    ProjectManager().validate_project(str(out))


if __name__ == "__main__":
    pack(Path(sys.argv[1]))
