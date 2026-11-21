import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from duel import apply_duels
from journal import long_answers, normalize_rows, read_jsonl
from team_protocol import identity_family, team_journal


def slot(identifier, prompt, kind="long"):
    return SimpleNamespace(id=identifier, prompt=prompt, kind=kind)


def row(endpoint, identifier, prompt, response):
    return dict(endpoint_id=endpoint, prompt_id=identifier, prompt=prompt, response=response)


class JournalTests(unittest.TestCase):
    def test_partial_snapshot_skips_only_a_torn_final_line(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "answers.jsonl"
            source.write_text('{"first": true}\n{"torn":', encoding="utf-8")
            self.assertEqual(list(read_jsonl(source, allow_incomplete_tail=True)), [{"first": True}])
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                list(read_jsonl(source))
            source.write_text('{"first": true}\n{"torn":\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                list(read_jsonl(source, allow_incomplete_tail=True))

    def test_four_field_reserve_repair(self):
        slots = [slot("long-01", "original"), slot("reserve-01", "reserve", "burst")]
        rows, endpoints = normalize_rows([
            row("ep-01", "long-01", "original", ""),
            row("ep-01", "reserve-01", "original", "recovered"),
        ], slots)
        self.assertEqual(endpoints, ["ep-01"])
        self.assertEqual(long_answers(rows, slots)[0], {"ep-01": {"long-01": "recovered"}})

    def test_append_only_metadata_does_not_count_as_a_response(self):
        slots = [slot("long-01", "prompt")]
        start = dict(event="start", endpoint="ep-01", id="long-01", prompt="prompt")
        with self.assertRaisesRegex(ValueError, "no completed"):
            normalize_rows([start], slots)
        answer = dict(start, event="answer", response="text", body={"ignored": True})
        rows, _ = normalize_rows([start, answer], slots)
        self.assertEqual(rows[0]["response"], "text")

    def test_shuffled_repairs_cover_distinct_slots_that_share_a_prompt(self):
        slots = [slot("burst-01", "same", "burst"), slot("burst-61", "same", "burst"),
                 slot("reserve-01", "one", "burst"), slot("reserve-02", "two", "burst")]
        rows, _ = normalize_rows([
            row("ep-01", "reserve-02", "same", "second"),
            row("ep-01", "burst-61", "same", ""),
            row("ep-01", "reserve-01", "same", "first"),
            row("ep-01", "burst-01", "same", ""),
        ], slots)
        by_id = {row["id"]: row["probe_id"] for row in rows}
        self.assertEqual(by_id["reserve-01"], "burst-01")
        self.assertEqual(by_id["reserve-02"], "burst-61")

    def test_conflicting_duplicate_is_rejected(self):
        slots = [slot("long-01", "prompt")]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            normalize_rows([
                row("ep-01", "long-01", "prompt", "first"),
                row("ep-01", "long-01", "prompt", "second"),
            ], slots)

    def test_same_family_duels_abstain_if_the_family_is_ambiguous(self):
        labels = {name: ("F1", name) for name in "abcdef"}
        duels = [dict(candidates=["a", "b"], plain_anchor="c"),
                 dict(candidates=["d", "e"], plain_anchor="f")]
        with patch("duel.load_replica", side_effect=AssertionError("must stay unloaded")):
            predicted, diagnostic = apply_duels(labels, [], [], duels)
        self.assertEqual(labels, predicted)
        self.assertEqual([row["status"] for row in diagnostic], ["multiple_duels_in_family"] * 2)

    def test_paired_ranking_requires_distinct_prompts_and_preserves_the_anchor(self):
        labels = {"a": ("F1", "S"), "b": ("F1", "B"), "c": ("F1", "S")}
        slots = [slot(f"long-{i:02d}", f"prompt {i}") for i in range(12)]
        raw = [row(ep, probe.id, probe.prompt, "valid") for ep in "ab" for probe in slots]
        entry = dict(candidates=["a", "b"], plain_anchor="c", checkpoint="local",
                     prompt_format="raw", temperature=1, top_p=1)
        rows, _ = normalize_rows(raw, slots)
        with patch("duel.load_replica"), patch("duel.rank_pair", return_value=1.0):
            predicted, report = apply_duels(labels, rows, slots, [entry])
        self.assertEqual(predicted["b"], labels["c"])
        self.assertEqual(predicted["c"], labels["c"])
        self.assertNotEqual(predicted["a"][1], labels["c"][1])
        self.assertEqual(report[0]["matched"], 12)
        slots[-1].prompt = slots[0].prompt
        with patch("duel.load_replica", side_effect=AssertionError("must stay unloaded")):
            predicted, report = apply_duels(labels, rows, slots, [entry])
        self.assertEqual(predicted, labels)
        self.assertEqual(report[0]["matched"], 11)


class TeamJournalTests(unittest.TestCase):
    def test_identity_needs_attribution_and_abstains_on_conflict(self):
        self.assertEqual(identity_family("I am a model developed by Meta AI."), "Llama")
        self.assertEqual(identity_family("I am Qwen2.5, a language model created by Alibaba Cloud."), "Qwen")
        self.assertEqual(identity_family("I am an AI assistant created by Anthropic."), "Anthropic")
        self.assertIsNone(identity_family("I was not created by Meta AI."))
        self.assertIsNone(identity_family("I'm a bot that can explain how Qwen works."))
        self.assertIsNone(identity_family("I am a model created by Meta or Alibaba."))

    def test_partial_team_journal_keeps_missing_targets_and_canonicalizes_longs(self):
        source = [dict(endpoint="ep-01", id="L1", repeat=0, prompt="fixed prompt", response="answer")]
        protocol = SimpleNamespace(Slot=lambda identifier, kind, wave, prompt: slot(identifier, prompt, kind))
        rows, endpoints, slots = team_journal(source, [], protocol, allow_partial=True,
                                              expected=["ep-01", "ep-02"])
        self.assertEqual(len(slots), 33)
        self.assertEqual(rows[0]["id"], "long-01")
        self.assertEqual(endpoints, ["ep-01", "ep-02"])
        with self.assertRaisesRegex(ValueError, "unfinished"):
            team_journal(source, [], protocol, allow_partial=False, expected=endpoints)


if __name__ == "__main__":
    unittest.main()
