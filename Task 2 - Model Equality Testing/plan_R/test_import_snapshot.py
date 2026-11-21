import unittest

from features import signatures
from import_snapshot import exact_flag_labels, project
from protocol import fingerprint, manifest


class ImportSnapshotTests(unittest.TestCase):
    def test_project_validates_prompt_and_preserves_incompatible_format_probe(self):
        answers = [{"endpoint": "ep-01", "id": src, "prompt": src, "repeat": 0, "response": text}
                   for src, text in (("F1", "hello"), ("F3", "captain"), ("F4", "2,3,5"), ("L01", "story"))]
        rows = list(project(answers, answers))
        self.assertEqual(rows[0]["sha256"], fingerprint(manifest()))
        self.assertEqual([row["id"] for row in rows[1:]], ["flag-01", "flag-03", "F4", "long-01"])
        sig = signatures(rows)["ep-01"]
        self.assertTrue(sig.flags["identity"])
        self.assertFalse(sig.flags["format"])
        self.assertIn("long-01", sig.long_by_prompt)
        with self.assertRaisesRegex(ValueError, "unexpected or duplicate"):
            list(project([dict(answers[0], prompt="different")], answers))
        with self.assertRaisesRegex(ValueError, "unexpected or duplicate"):
            list(project([answers[0], answers[0]], answers))

    def test_exact_flags_require_both_texts_and_do_not_cross_families(self):
        labels = {"a": ("F1", "m1"), "b": ("F1", "m2"), "c": ("F1", "m3"),
                  "d": ("F2", "m4"), "e": ("F1", "m5")}
        texts = {"a": ("calm", "2,3"), "b": ("calm", "2,3"), "c": ("calm", "2, 3"),
                 "d": ("calm", "2,3"), "e": ("calm", "")}
        rows = [{"endpoint": ep, "source_id": probe, "response": text}
                for ep, pair in texts.items() for probe, text in zip(("F3", "F4"), pair)]
        self.assertEqual(exact_flag_labels(rows, labels), dict(labels, b=("F1", "m1")))
