import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

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
