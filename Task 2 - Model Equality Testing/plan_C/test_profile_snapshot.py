import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import profile_snapshot as p


class ProfileTests(unittest.TestCase):
    def test_continuations_and_language(self):
        prompt = "Continue the following text naturally:\n\nUne histoire dans les bois"
        self.assertEqual(p.expected_language(prompt), "fr")
        self.assertEqual(p.generated("**Une histoire dans les bois continue.", prompt), "continue.")
        self.assertEqual(p.generated("La suite.", prompt), "La suite.")
        self.assertEqual(p.divergence(["a", "b"], ["a", "c"]), 2)
        self.assertIsNone(p.divergence(["a"], ["a"]))

    def test_punctuated_stem_and_word_boundary(self):
        prompt = "Continue the following text:\n\nOnce upon a time"
        self.assertEqual(p.generated("Once upon a time, the sun rose.", prompt + ","), "the sun rose.")
        self.assertEqual(p.generated("Once upon a timetable", prompt), "Once upon a timetable")

    def test_no_rank_or_diversity_in_tiny_or_short_samples(self):
        self.assertIsNone(p.rank(.5, [.1, .2, .3]))
        self.assertEqual(p.rank(.5, [.5] * 7), .5)
        self.assertIsNone(p.features("short.", "en")["div"])
        pool = {e: {"L01": "red " * 64} for e in ("ep-01", "ep-04", "ep-12", "ep-14")}
        self.assertFalse(p.compare(pool, {"L01": "Colors?"}))
        pool["ep-17"] = {"L01": "blue " * 64}
        r = p.compare(pool, {"L01": "Colors?"})["ep-17"][0]
        self.assertEqual((r["rare"], r["divergence"], r["peers"]), (1., 1, 4))
        self.assertFalse(r["temp_vote"])

    def test_partial_disguises_and_confirmed_singleton(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            infile, out, table, proofs = (folder / f for f in ("input", "out", "table", "proofs"))
            def record(i, text):
                return json.dumps(dict(endpoint="ep-01", id="F1", prompt="Who?", repeat=i, response=text))
            infile.write_text(record(1, "I am a chef") + "\n" + record(0, "I am a journalist"))
            argv = ["run", str(infile), "-o", str(out), "--table", str(table)]
            with patch("sys.argv", argv):
                p.main()
            def read(path):
                with path.open() as f:
                    return list(csv.DictReader(f))
            self.assertEqual(len(read(out)), 33)
            self.assertEqual(len({r["model"] for r in read(out)}), 3)
            self.assertEqual(read(table)[0]["F1_F3_role"], "journalist")
            self.assertEqual(read(table)[1]["etiquette"], "inconnu")
            proofs.write_text("endpoint,kind,evidence\nep-02,fp8,synthetic known weights\n")
            with patch("sys.argv", argv + ["--confirmed-singletons", str(proofs)]):
                p.main()
            self.assertEqual(read(out)[1]["model"], "QB-ep-02")
            self.assertEqual(len({r["model"] for r in read(out)}), 4)


if __name__ == "__main__":
    unittest.main()
