import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from protocol import fingerprint, manifest
from submit import main


class SubmitTests(unittest.TestCase):
    def test_rejects_output_aliases_of_the_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "answers.jsonl"
            journal.write_text("untouched\n")
            symlink = Path(tmp) / "symlink.csv"
            symlink.symlink_to(journal)
            hardlink = Path(tmp) / "hardlink.csv"
            hardlink.hardlink_to(journal)
            for output in (journal, symlink, hardlink):
                with self.subTest(output=output), patch("sys.argv", ["submit.py", str(journal), str(output)]):
                    with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                        main()
                    self.assertIn("output must not overwrite the input journal", err.getvalue())
                    self.assertEqual(journal.read_text(), "untouched\n")

    def test_final_export_requires_all_slots_to_finish(self):
        slots = manifest()
        wave_one = [s for s in slots if s.wave == 1]
        header = {"event": "manifest", "sha256": fingerprint(slots)}
        with tempfile.TemporaryDirectory() as tmp:
            journal, output = Path(tmp) / "answers.jsonl", Path(tmp) / "submission.csv"
            base = ["submit.py", str(journal), str(output), "--target", "refs"]
            for complete, partial in ((False, False), (False, True), (True, False)):
                with self.subTest(complete=complete, partial=partial):
                    rows = [header, *(
                        {"endpoint": f"ref-{i:02d}", "event": "rejected" if s.id == "reserve-04" else "answer",
                         "id": s.id, "response": "test"}
                        for i in range(1, 10) for s in (slots if complete else wave_one)
                    )]
                    journal.write_text("\n".join(map(json.dumps, rows)))
                    with patch("sys.argv", base + (["--allow-partial", "--provisional"] if partial else [])):
                        if complete or partial:
                            main()
                            self.assertEqual(len(output.read_text().splitlines()), 10)
                            output.unlink()
                        else:
                            with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                                main()
                            self.assertIn("collection is incomplete (774 unfinished slots)", err.getvalue())
                            self.assertFalse(output.exists())
