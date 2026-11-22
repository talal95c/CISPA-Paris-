"""Compatibility entry point for the original experiment commands."""
from pathlib import Path
from runpy import run_path

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "Task 2 - Model Equality Testing" / "score_wb.py"
    run_path(str(target), run_name="__main__")
