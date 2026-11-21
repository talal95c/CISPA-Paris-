import csv
import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from collect import TASK_PATH, completion, request
from features import DEFAULTS, clipped_llr, dispersion, flag_features, parse_burst, signatures
from partition import confusion, family_proposals, mcc, optimize_joint, pairs, rule, score
from phase import effectiveness, free_text_null, main as phase_main, watermark_gate
from protocol import manifest
from replay import check_budget, holdout
from simulate import VARIANTS, phase_jobs, replay_slots
from simulate_server import make_handler
from submit import FROZEN_CONFIG, write_csv


class AnalysisTests(unittest.TestCase):
    def test_burst_alignment_exact_repeats_and_duplicate_rejection(self):
        items = manifest()[34].items
        tag, other = items[0].tag, items[1].tag
        self.assertEqual(parse_burst(f"{tag} A\n{tag} B\n{other} **B**", items), {other: 1})
        a = f"{tag} A\n{other} B"
        b = f"{tag} B\n{other} B"
        self.assertEqual(dispersion([(a, b, items)])["disagreement"], .5)
        self.assertTrue(dispersion([(a, a, items), (b, b, items)])["greedy"])

    def test_format_alone_is_not_an_identity_flag_and_reserve_repairs(self):
        self.assertEqual(flag_features({"flag-04": "not ok"})["strength"], 0)
        rows = [
            {"event": "answer", "endpoint": "ref-01", "id": "flag-01", "response": ""},
            {"event": "answer", "endpoint": "ref-01", "id": "reserve-01", "probe_id": "flag-01",
             "response": "I am Captain Brine, a pirate."},
        ]
        self.assertTrue(signatures(rows)["ref-01"].flags["identity"])
        self.assertEqual(clipped_llr((None, None, None), DEFAULTS["family"]), 0)
        self.assertEqual(clipped_llr((-10, -10, -10), DEFAULTS["family"], (1, 1, 1)), 3)

    def test_structural_k_and_joint_mcc(self):
        core = {"a", "b"}
        self.assertEqual(rule(core, set()), core)
        self.assertEqual(rule(core, {"c"}), {"a", "b", "c"})
        self.assertEqual(rule(core, {"c", "d"}), {"a", "b", "c", "d"})
        self.assertEqual(rule(core, {"c", "d", "e"}), core)
        full = ("a", "b", "c")
        evidence = {("a", "c"): 3, ("b", "c"): 3}
        self.assertEqual(family_proposals(full, evidence, core, {})[0], [frozenset(full)])
        self.assertEqual(family_proposals(full, evidence, core, {}, candidate_floor=4)[0], [frozenset(core)])
        groups = [("a", "b", "c"), ("d", "e", "f")]
        guesses = [[frozenset(("a", "b")), frozenset(groups[0])],
                   [frozenset(("d", "e")), frozenset(groups[1])]]
        worlds = [{pairs("ab"): .4, pairs("abc"): .6}, {pairs("de"): .7, pairs("def"): .3}]
        expected, joint = optimize_joint(groups, list(zip(guesses, worlds)))
        manual = []
        for a in guesses[0]:
            for b in guesses[1]:
                manual.append((sum(p*q*mcc(*confusion(x | y, pairs(a) | pairs(b), 6))
                                   for x, p in worlds[0].items() for y, q in worlds[1].items()), [a, b]))
        self.assertAlmostEqual(expected, max(manual)[0])
        self.assertEqual(joint, max(manual)[1])

    def test_official_csv_and_model_score_uses_true_families(self):
        truth = {"ref-01": ("A", "A1"), "ref-02": ("A", "A1"), "ref-03": ("A", "A2"),
                 "ref-04": ("B", "B1")}
        guess = dict(truth, **{"ref-04": ("A", "A1")})
        result = score(truth, guess)
        self.assertAlmostEqual(result["model"], 1.0)
        self.assertEqual(result["family"], 0)
        self.assertEqual(result["total"], .9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "practice.csv"
            write_csv(path, truth)
            with path.open(newline="") as handle:
                self.assertEqual(list(csv.reader(handle))[0], ["endpoint", "family", "model"])

    def test_phase_statistics_dont_promote_missing_text(self):
        missing = {"a": np.full((20, 2), np.nan)}
        self.assertEqual(effectiveness(missing), (None, 0))
        self.assertEqual(watermark_gate(missing, missing)["verdict"], "non mesurable")
        varied = {"a": np.asarray([[0, 1], [0, 0], [1, 1], [1, 0]])}
        self.assertEqual(effectiveness(varied), (1.0, 4))
        texts = [{"id": "a", "repeat": i, "response": value} for i, value in enumerate(("apple", "zebra"))]
        self.assertEqual(free_text_null(texts), {"text_pairs": 1, "text_mean": 1.0, "text_q95": 1.0})

    def test_phase_reads_common_phase_probes_and_heldout_replay_probes(self):
        dev_burst = next(slot for slot in replay_slots(0) if slot.id == "burst-01")
        dev_tags = {item.tag for item in dev_burst.items}
        held = {slot.id: slot for slot in replay_slots(0, heldout=True)}
        item = next(item for item in held["burst-01"].items if item.tag not in dev_tags)
        rows = [{"event": "simulation", "items": 20, "heldout": True}]
        for slot_id, choice in (("burst-01", "A"), ("burst-61", "B")):
            rows.append({"event": "answer", "endpoint": "ref-09", "block": 0, "id": slot_id,
                         "variant": "temp", "response": f"{item.tag} {choice}"})
        block, slot, rep = next(phase_jobs("plain", 1))
        rows.append({"event": "answer", "endpoint": "ref-01", "block": block, "id": slot.id,
                     "variant": "plain", "repeat": rep, "response": f"{slot.items[0].tag} A"})
        with tempfile.TemporaryDirectory() as tmp:
            journal, output = Path(tmp) / "heldout.jsonl", Path(tmp) / "phase.json"
            journal.write_text("\n".join(map(json.dumps, rows)))
            with patch("sys.argv", ["phase.py", str(journal), "--output", str(output)]), redirect_stdout(io.StringIO()):
                phase_main()
            result = json.loads(output.read_text())
        self.assertEqual(result["B7"]["n_positives"], 1)
        self.assertEqual(result["20"]["B2"]["n"], 1)
        self.assertAlmostEqual(result["20"]["B3"]["plain_item_coverage"], 1 / len(slot.items))

    def test_exact_budget_and_stable_holdout(self):
        slots = replay_slots(3)
        rows = [{"endpoint": str(ep), "id": slot.id, "prompt": slot.prompt}
                for ep in range(2*len(VARIANTS)) for slot in slots]
        check_budget(rows, slots)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            check_budget(rows + [rows[0]], slots)
        items = [item for slot in slots for item in slot.items]
        self.assertTrue(all(holdout(i) for i in items if i.type == 5))
        self.assertTrue(any(not holdout(i) for i in items))

    def test_local_model_loopback_api(self):
        model = Mock()
        model.generate.return_value = iter([("mock text", 2)])
        with ThreadingHTTPServer(("127.0.0.1", 0), make_handler(model)) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}{TASK_PATH}"
                status, _, listing = request(base, "/models", "mock")
                self.assertEqual((status, len(listing["data"])), (200, 9))
                payload = completion("ref-06", "test")
                status, _, reply = request(base, "/chat/completions", "mock", payload)
                self.assertEqual((status, reply["choices"][0]["message"]["content"]), (200, "mock text"))
                model.configure.assert_called_once_with("watermark")
                model.restore.assert_called_once()
                self.assertEqual(request(base, "/chat/completions", "mock", dict(payload, max_tokens=257))[0], 400)
            finally:
                server.shutdown()
                thread.join()

    def test_frozen_families_and_full_prompt_splits_are_disjoint(self):
        cfg = json.loads(FROZEN_CONFIG.read_text())
        self.assertTrue(set(cfg["development_families"]).isdisjoint(cfg["heldout_families"]))
        dev, test = ({slot.prompt for block in range(cfg["blocks"]) for slot in replay_slots(block, 20, heldout)}
                     for heldout in (False, True))
        self.assertTrue(dev.isdisjoint(test))


if __name__ == "__main__":
    unittest.main()
