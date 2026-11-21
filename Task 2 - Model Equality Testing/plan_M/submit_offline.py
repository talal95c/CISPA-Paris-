"""Turn a local JSONL response journal into an official-format CSV."""

import argparse
import json
from pathlib import Path

from duel import apply_duels
from journal import load_shared, manifest_from_json, normalize_rows, read_jsonl


DEFAULT_SHARED = Path(__file__).resolve().parent.parent / "plan_R"


def predict(rows, endpoints, slots, features, partition, config, profile,
            *, duels=(), item_filter=None):
    signatures = features.signatures(rows, slots, item_filter=item_filter)
    labels, info = partition.infer(signatures, config["params"], config["profiles"][profile])
    missing = sorted(set(endpoints) - labels.keys())
    # An endpoint with no usable answer has no evidence for sharing a label.
    for index, endpoint in enumerate(missing, 1):
        family = f"unobserved-{index}"
        labels[endpoint] = family, family
    labels, evidence = apply_duels(labels, rows, slots, duels)
    return labels, {"families": info["families"], "unobserved": missing, "duels": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--output", type=Path, help="destination CSV")
    parser.add_argument("--shared-dir", type=Path, default=DEFAULT_SHARED, help="directory of the shared Plan R code")
    parser.add_argument("--target", choices=("scored", "refs", "replay"), default="scored")
    parser.add_argument("--items", type=int, choices=(12, *range(20, 26)), default=20)
    parser.add_argument("--manifest", type=Path, help="explicit JSON manifest exported by the common collector")
    parser.add_argument("--profile", choices=("R", "simple"), default="simple")
    parser.add_argument("--duels", type=Path, help="optional JSON list of already vetted white-box duels")
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count for optional white-box inference")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="check journal consistency without inference or output")
    args = parser.parse_args()
    if not (args.output or args.dry_run):
        parser.error("--output is required for CSV generation")
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        inputs = [path for path in (args.journal, args.manifest, args.duels) if path is not None]
        if args.output and any(args.output.resolve() == path.resolve() or
                               (args.output.exists() and path.exists() and args.output.samefile(path))
                               for path in inputs):
            raise ValueError("The output CSV must be separate from the input files")
        features, partition, protocol, submission = load_shared(args.shared_dir)
        slots = (manifest_from_json(args.manifest, protocol) if args.manifest
                 else protocol.manifest(size=args.items))
        source = list(read_jsonl(args.journal))
        if source and source[0].get("event") == "manifest" and "sha256" in source[0]:
            if source[0]["sha256"] != protocol.fingerprint(slots):
                raise ValueError("The collector journal uses a different manifest")
        prefix, count = ("ep", 33) if args.target == "scored" else ("ref", 9)
        expected = [f"{prefix}-{i:02d}" for i in range(1, count + 1)]
        rows, endpoints = normalize_rows(source, slots, allow_partial=args.allow_partial,
                                         expected=None if args.target == "replay" else expected)
        if args.dry_run:
            print(json.dumps({"endpoints": len(endpoints), "completed_slots": len(rows),
                              "manifest_slots_per_endpoint": len(slots)}))
            return
        config = json.loads(submission.FROZEN_CONFIG.read_text(encoding="utf-8"))
        duels = json.loads(args.duels.read_text(encoding="utf-8")) if args.duels else []
        if not isinstance(duels, list):
            raise ValueError("--duels must be a JSON list")
        if duels:
            import torch
            torch.set_num_threads(args.threads)
        labels, details = predict(rows, endpoints, slots, features, partition, config, args.profile, duels=duels)
        submission.write_csv(args.output, labels)
        print(json.dumps({"endpoints": len(labels), **details}, ensure_ascii=False))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
