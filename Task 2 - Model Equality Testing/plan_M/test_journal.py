import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from duel import apply_duels
from journal import long_answers, normalize_rows, read_jsonl
from long_matches import partition_long_matches, prompt_matches
from team_protocol import FLAGS, identity_family, split_flag_families, team_journal


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
            source.write_bytes(b'{"first": true}\n{"torn": "\xc3')
            self.assertEqual(list(read_jsonl(source, allow_incomplete_tail=True)), [{"first": True}])
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                list(read_jsonl(source))

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
    def test_flag_families_require_two_exact_answers_within_one_anchor(self):
        labels = {ep: ("Qwen", "shared") for ep in ("a", "b", "c", "d")}
        labels["e"] = ("Llama", "other")
        flags = {
            "a": ("Helpful.", "2,3,5"),
            "b": ("Helpful.\n", "2,3,5"),
            "c": ("Helpful.", "2, 3, 5"),
            "d": ("Helpful.", ""),
            "e": ("Helpful.", "2,3,5"),
        }
        rows = [dict(endpoint=ep, id=FLAGS[probe], event="answer", response=text)
                for ep, texts in flags.items() for probe, text in zip(("F3", "F4"), texts)]
        split = split_flag_families(labels, rows)
        self.assertEqual(split["a"], split["b"])
        self.assertNotEqual(split["a"][0], labels["a"][0])
        self.assertNotEqual(split["a"][1], labels["a"][1])
        for ep in "cde":
            self.assertEqual(split[ep], labels[ep])

    def test_identity_recognizes_explicit_name_and_bot_creator(self):
        self.assertEqual(identity_family("I am a language model, known as Qwen."), "Qwen")
        self.assertEqual(identity_family("I'm a bot developed by Meta AI."), "Llama")
        self.assertEqual(identity_family(
            "I am an AI assistant and a language model developed by Anthropic."
        ), "Anthropic")
        self.assertEqual(identity_family(
            "I am an AI assistant and a language model, known as Qwen."
        ), "Qwen")
        self.assertEqual(identity_family(
            "I’m an AI assistant and an open-source language model developed by Alibaba Cloud."
        ), "Qwen")
        self.assertEqual(identity_family(
            "I’m an AI assistant and a language model known as Qwen2.5-Instruct."
        ), "Qwen")
        self.assertEqual(identity_family(
            "I am an AI assistant and a language model developed by Alibaba Cloud to help users."
        ), "Qwen")
        self.assertEqual(identity_family(
            "I am an AI assistant and a large language model created by Meta AI, and I can help."
        ), "Llama")
        self.assertEqual(identity_family(
            "I'm an AI assistant and a language model known as Claude 3.5 Sonnet."
        ), "Anthropic")
        self.assertIsNone(identity_family(
            "I’m an AI assistant and a chatbot known as Claude Sonnet also answers questions."
        ))
        self.assertIsNone(identity_family("I'm a bot that can explain a model known as Qwen."))
        self.assertIsNone(identity_family("I'm a bot and can explain a model known as Qwen."))
        self.assertIsNone(identity_family(
            "I'm an AI assistant and the platform also offers another chatbot developed by Anthropic."
        ))
        self.assertIsNone(identity_family(
            "I’m an AI assistant and the platform offers an open-source chatbot developed by Anthropic."
        ))
        self.assertIsNone(identity_family(
            "I’m an AI assistant and an open-source chatbot developed by Anthropic can answer that too."
        ))
        self.assertIsNone(identity_family(
            "I’m an assistant and a language model known as Qwen also answers questions."
        ))

    def test_identity_needs_attribution_and_abstains_on_conflict(self):
        self.assertEqual(identity_family("I am a model developed by Meta AI."), "Llama")
        self.assertEqual(identity_family("I am Qwen2.5, a language model created by Alibaba Cloud."), "Qwen")
        self.assertEqual(identity_family("I am an AI assistant created by Anthropic."), "Anthropic")
        self.assertIsNone(identity_family("I was not created by Meta AI."))
        self.assertIsNone(identity_family("I'm a bot that can explain how Qwen works."))
        self.assertIsNone(identity_family("I am a model created by Meta or Alibaba."))
        self.assertIsNone(identity_family("I am unable to disclose my identity. Qwen was developed by Alibaba Cloud."))
        self.assertIsNone(identity_family("This AI model is known as Qwen."))

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

    def test_long_matches_require_evidence_for_every_pair(self):
        slots = [slot(f"long-{i}", f"prompt {i}") for i in range(1, 5)]
        samples = {probe.id: probe.prompt * 25 for probe in slots}
        answers = {
            "a": {"long-1": samples["long-1"], "long-2": samples["long-2"]},
            "b": {**samples, "long-2": samples["long-2"] + " continued"},
            "c": {"long-3": samples["long-3"], "long-4": samples["long-4"]},
            "d": {"long-1": samples["long-1"], "long-2": samples["long-2"]},
        }
        rows, _ = normalize_rows([
            row(ep, pid, next(probe.prompt for probe in slots if probe.id == pid), text)
            for ep, texts in answers.items() for pid, text in texts.items()
        ], slots, allow_partial=True)
        labels = {ep: ("F2" if ep == "d" else "F1", ep) for ep in answers}
        actual, edges = partition_long_matches(labels, rows, slots)
        self.assertEqual(actual["a"], actual["b"])
        self.assertNotEqual(actual["a"], actual["c"])
        self.assertNotEqual(actual["a"], actual["d"])
        self.assertEqual(edges[0]["matches"], [("long-1", "exact"), ("long-2", "prefix")])
        self.assertEqual([edge["endpoints"] for edge in edges], [["a", "b"], ["b", "c"]])

    def test_repeated_identical_output_only_counts_once(self):
        slots = {"long-1": slot("long-1", "first"), "long-2": slot("long-2", "second")}
        boilerplate = {"long-1": "A" * 140, "long-2": "A" * 140}
        self.assertEqual(prompt_matches(boilerplate, boilerplate, slots), [("long-1", "exact")])

    def test_one_endpoint_repeating_output_cannot_supply_a_second_match(self):
        slots = {"long-1": slot("long-1", "first"), "long-2": slot("long-2", "second")}
        common = "A" * 140
        left = {"long-1": common + " continued", "long-2": common + " continued"}
        right = {"long-1": common + " continued", "long-2": common}
        self.assertEqual(prompt_matches(left, right, slots), [("long-1", "exact")])
        self.assertEqual(prompt_matches(right, left, slots), [("long-1", "exact")])
        # Both endpoints vary their replies, but the shared prefix is the same.
        left["long-1"] = common
        right["long-1"] = common + " followed"
        self.assertEqual(prompt_matches(left, right, slots), [("long-1", "prefix")])

    def test_duplicate_prompt_uses_an_alternative_when_first_match_repeats(self):
        slots = {f"long-{i}": slot(f"long-{i}", prompt)
                 for i, prompt in enumerate(["first", "second", "first"], 1)}
        answers = {"long-1": "A" * 140, "long-2": "A" * 140, "long-3": "B" * 140}
        self.assertEqual(prompt_matches(answers, answers, slots), [("long-2", "exact"), ("long-3", "exact")])


if __name__ == "__main__":
    unittest.main()
