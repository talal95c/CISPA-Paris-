"""Generate the official endpoint,family,model CSV from an offline journal."""
import argparse
import csv
import json
from pathlib import Path

from features import read_rows, signatures
from partition import infer
from protocol import fingerprint, manifest


FROZEN_CONFIG = Path(__file__).with_name("frozen_r.json")


def write_csv(output, labels):
    with output.open("w", encoding="utf-8", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["endpoint", "family", "model"])
        out.writerows((ep, *labels[ep]) for ep in sorted(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", choices=("scored", "refs"), default="scored")
    parser.add_argument("--items", type=int, default=20, choices=[12, *range(20, 26)])
    parser.add_argument("--config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--profile", choices=("R", "simple"), default="R")
    parser.add_argument("--provisional", action="store_true", help="S1: group flags but leave plain candidates separate")
    args = parser.parse_args()
    if args.output.resolve() == args.journal.resolve() or (
        args.output.exists() and args.journal.exists() and args.output.samefile(args.journal)
    ):
        parser.error("output must not overwrite the input journal")
    rows = read_rows(args.journal)
    slots = manifest(size=args.items)
    if not rows or rows[0].get("sha256") != fingerprint(slots):
        parser.error("journal manifest does not match --items")
    config = json.loads(args.config.read_text())
    sigs = signatures(rows, slots)
    prefix, count = ("ep", 33) if args.target == "scored" else ("ref", 9)
    expected = {f"{prefix}-{i:02d}" for i in range(1, count + 1)}
    if set(sigs) != expected:
        parser.error(f"journal endpoints differ from target (missing {sorted(expected - sigs.keys())}, "
                     f"unexpected {sorted(sigs.keys() - expected)})")
    settings = config["profiles"][args.profile]
    if args.provisional:
        settings = dict(settings, candidate_floor=4, temp_threshold=None)
    labels, info = infer(sigs, config["params"], settings)
    write_csv(args.output, labels)
    print(f"{len(labels)} endpoints, {len(info['families'])} families: {args.output}")


if __name__ == "__main__":
    main()
