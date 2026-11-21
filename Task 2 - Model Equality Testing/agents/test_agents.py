import tempfile
import unittest
from pathlib import Path

import arena
from control import control
from check_candidate import official, audit
from famou.evaluator import evaluate, read_expr


class AgentsTest(unittest.TestCase):
    def test_increasing_control_budget_keeps_previous_trials(self):
        self.assertEqual(control(4)["attempts"], control(9)["attempts"][:4])

    def test_audit_refuses_missing_family_or_block(self):
        for held in (
            {},
            {"holdout3": [{}] * 5},
            {"holdout3": [{}] * 5, "holdout4": [{}] * 4},
        ):
            with self.assertRaisesRegex(ValueError, "five blocks each"):
                audit("lex", 1.0, held, None, None)

    def test_no_candidate_execution(self):
        for expr in (
            "obj.x",
            'open("labels")',
            "lex**4",
            "min(lex)",
            "max(*[])",
            "1e999",
            "(lambda:1)()",
        ):
            with self.assertRaises(ValueError):
                arena.formula(expr)
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "candidate.py"
            path.write_text('EXPR = "lex"\nraise RuntimeError("must never run")')
            self.assertEqual(evaluate(path)["validity"], 0)
            path.write_text('EXPR = "min(lex,shape)+0.5*length"')
            self.assertEqual(arena.formula(read_expr(path))((1, 1, 2, 4)), 3)

    def test_tied_heights_are_one_level(self):
        cohort = dict(labels=[0, 0, 1], features=[[1, 0, 0, 0]] * 3)
        cuts = arena.states(cohort, arena.formula("lex"))
        self.assertEqual(cuts, [(-1.0, (0, 0, 1, 2)), (1.0, (1, 2, 0, 0))])

    def test_perfect_separation(self):
        cohort = dict(labels=[0, 0, 1], features=[[x, 0, 0, 0] for x in (0.1, 2, 3)])
        result = arena.evaluate("lex", {"a": [cohort], "b": [cohort]})
        self.assertEqual(
            result, dict(mcc_by_family=[1.0, 1.0], objective=1.0, cutoff=0.1)
        )
        self.assertEqual(arena.at([arena.states(cohort, arena.formula("lex"))], -1), 0)

    def test_scoring_uses_true_families(self):
        truth = dict(a=("A", "1"), b=("A", "1"), c=("A", "2"), d=("B", "3"))
        self.assertEqual(official(truth, truth), dict(family=1.0, model=1.0, score=1.0))
        pred = {k: (fam, k) for k, (fam, _) in truth.items()}
        self.assertEqual(official(truth, pred), dict(family=1.0, model=0.0, score=0.1))
        self.assertEqual(official(truth, {**truth, "d": ("A", "1")})["model"], 1.0)

    def test_aide_policy_keeps_parent(self):
        try:
            from aide_cpu import search
        except ModuleNotFoundError as exc:
            if exc.name != "aide":
                raise
            self.skipTest("Install the AIDE runtime")
        proposals = iter(("prefix", "bad_name"))
        result = search("lex", lambda parent, hist: next(proposals), 2, arena.load())
        self.assertEqual(result["journal"][1]["parent"], 0)
        self.assertIn("error", result["journal"][2])
