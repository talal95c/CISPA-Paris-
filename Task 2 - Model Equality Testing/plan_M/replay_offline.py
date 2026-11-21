"""Evaluate the offline pipeline on the frozen, exact-budget Plan R journals."""

import argparse
import json
from pathlib import Path

from duel import MIN_MATCHED_LONGS, load_replica, matched_longs, rank_pair
from journal import load_shared, long_answers, normalize_rows
from submit_offline import DEFAULT_SHARED, predict


FROZEN_DUELS = {
    "development": (
        ("ref", "smollm2-135m-instruct", "chat"),
        ("sim", "tinystories-33m", "user-assistant"),
    ),
    "holdout": (
        ("hold-a", "pythia-14m", "user-assistant"),
        ("hold-b", "tinystories-8m", "user-assistant"),
    ),
}


def simulated_duels(split, checkpoint_root):
    return [
        dict(candidates=[f"{prefix}-01", f"{prefix}-02"],
             plain_anchor=f"{prefix}-03", checkpoint=str(checkpoint_root / directory),
             prompt_format=style, temperature=0.65, top_p=1.0, scheme="w8a16")
        for prefix, directory, style in FROZEN_DUELS[split]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journals", nargs=2, type=Path)
    parser.add_argument("--shared-dir", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--split", required=True, choices=tuple(FROZEN_DUELS))
    parser.add_argument("--checkpoint-root", type=Path, help="also test with sim-known candidate identities")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    features, partition, _, submission = load_shared(args.shared_dir)
    import replay
    from simulate import replay_slots

    heldout = args.split == "holdout"
    config = json.loads(submission.FROZEN_CONFIG.read_text(encoding="utf-8"))
    blocks = replay.load_blocks(args.journals, heldout)
    if sorted(blocks) != list(range(config["blocks"])):
        raise ValueError("Expected five exact-budget blocks")
    duels = simulated_duels(args.split, args.checkpoint_root) if args.checkpoint_root else []
    if duels:
        import torch
        torch.set_num_threads(args.threads)
    results, orientations = [], []
    for block, source in sorted(blocks.items()):
        slots = replay_slots(block, config["items"], heldout)
        replay.check_budget(source, slots)
        # Inference receives exactly the four requested journal fields.
        rows, endpoints = normalize_rows((
            dict(endpoint_id=row["endpoint"], prompt_id=row["id"],
                 prompt=row["prompt"], response=row["response"])
            for row in source
        ), slots)
        truth = {ep: answer for ep, (answer, _) in replay.truth_rows(source).items()}
        gate = lambda item: replay.holdout(item) == heldout
        result = {"block": block}
        for profile in ("simple", "R"):
            labels, _ = predict(rows, endpoints, slots, features, partition, config, profile, item_filter=gate)
            result[profile] = partition.score(truth, labels)
        if duels:
            labels, detail = predict(rows, endpoints, slots, features, partition, config, "simple",
                                     duels=duels, item_filter=gate)
            result["known_duels"] = partition.score(truth, labels)
            result["duel_status"] = detail["duels"]
            answers, longs = long_answers(rows, slots)
            for duel, status in zip(duels, detail["duels"], strict=True):
                left, right = duel["candidates"]
                matched = matched_longs(answers, longs, left, right)
                margin = status.get("margin_nats")
                if (status["status"] in {"different_families", "multiple_duels_in_family"}
                        and len(matched) >= MIN_MATCHED_LONGS):
                    margin = rank_pair(load_replica(duel["checkpoint"], duel["scheme"]), answers, matched,
                                       left, right, duel["prompt_format"], duel["temperature"], duel["top_p"])
                orientations.append(dict(block=block, candidates=[left, right], matched=len(matched),
                                         margin_nats=margin, correct=margin is not None and margin < 0))
        results.append(result)
    profiles = ["simple", "R", "known_duels"] if duels else ["simple", "R"]
    print(json.dumps({
        "split": args.split,
        "blocks": results,
        "within_family_orientations_with_known_candidates": orientations,
        "summary": {profile: replay.summary([row[profile] for row in results]) for profile in profiles},
    }, indent=2))


if __name__ == "__main__":
    main()
