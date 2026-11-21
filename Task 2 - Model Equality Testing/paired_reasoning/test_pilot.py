import unittest
from unittest.mock import patch

import pilot


class PilotTest(unittest.TestCase):
    def setUp(self):
        self.items = {"test": {"gold": "7", "a": "A?", "b": "B?"}}
        self.db = pilot.connect(":memory:", self.items)
        self.addCleanup(self.db.close)

    def fill(self):
        rows = [(*j, "7" if j[-1] in pilot.BASE else "5", "stop")
                for b in (1, 2) for j in pilot.jobs(self.items, b)]
        self.db.executemany("INSERT INTO responses VALUES(?,?,?,?,?,?,?)", rows)

    def test_unparseable_numbers_abstain(self):
        for noise in ("", "2.5", "7 or 8", "2/3", "I considered 7 items", "1,003"):
            self.assertIsNone(pilot.parse(noise))
        self.assertEqual(pilot.parse("3 + 4 = 7\n**Final answer: 7**"), 7)
        self.assertEqual(pilot.parse("7."), 7)

    def test_same_wrong_number_must_repeat(self):
        self.fill()
        self.db.execute("UPDATE responses SET answer='4' WHERE block=2 AND ref='ref-02'")
        result = pilot.analyze(self.db, self.items)["items"][0]
        self.assertEqual(set(result["confirmed"]), set(pilot.VARIANTS) - {"ref-02"})

    def test_disguises_and_truncation_do_not_help(self):
        self.fill()
        self.db.execute("UPDATE responses SET finish='length' WHERE block=2 AND ref='ref-05' AND form='a'")
        result = pilot.analyze(self.db, self.items)
        self.assertEqual(result["usable"], result["responses"] - 2)
        self.assertFalse(result["items"][0]["confirmed"])

    def test_no_incomplete_audit(self):
        with self.assertRaises(ValueError):
            pilot.analyze(self.db, self.items)
        with self.assertRaises(ValueError):
            pilot.collect(self.db, self.items, 2, "")

    def test_no_scored_endpoints_and_resume(self):
        with self.assertRaises(ValueError):
            pilot.query("ep-01", "", "")
        with patch.object(pilot, "query", return_value=("7", "stop")) as mock:
            pilot.collect(self.db, self.items, 1, "")
            pilot.collect(self.db, self.items, 1, "")
        self.assertEqual(mock.call_count, 36)
        self.assertTrue(all(call.args[0] in pilot.REFS for call in mock.call_args_list))

    def test_http_is_refused_before_request(self):
        with patch.dict("os.environ", CISPA_HTTPS_ORIGIN="http://34.147.229.55"):
            with self.assertRaises(ValueError):
                pilot.query("ref-01", "test", "dummy")

    def test_retry_after(self):
        self.assertEqual(pilot.retry_seconds("19", 0), 19)
        date = "Thu, 01 Jan 1970 00:01:00 GMT"
        self.assertEqual(pilot.retry_seconds(date, 12), 48)
        self.assertEqual(pilot.retry_seconds(date, 65), 0)
        self.assertEqual(pilot.retry_seconds("invalid", 0), 0)
