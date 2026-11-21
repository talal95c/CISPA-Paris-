"""FaMou evaluation contract, restricted to declarative candidates."""

import ast
import json
from pathlib import Path
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if (HERE / "arena.py").exists() else HERE.parent))
import arena


def read_expr(path):
    path = Path(path)
    if path.stat().st_size > 2048:
        raise ValueError("Candidate file too large")
    match ast.parse(path.read_text()).body:
        case [
            ast.Assign(
                targets=[ast.Name(id="EXPR")], value=ast.Constant(value=str() as s)
            )
        ]:
            arena.formula(s)
            return s
    raise ValueError("Expected exactly one EXPR string literal assignment")


def evaluate(
    path_user_py: str, task_name: str = "default", timeout: int = 3600
) -> dict:
    start = monotonic()
    try:
        score = arena.evaluate(read_expr(path_user_py), arena.load())
        result = dict(
            validity=1.0,
            combined_score=(1 + score["objective"]) / 2,
            error_info="",
            **score,
        )
    except (ValueError, TypeError, SyntaxError, OSError) as exc:
        result = dict(validity=0.0, combined_score=0.0, error_info=str(exc))
    return dict(result, cost_time=monotonic() - start)


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1])))
