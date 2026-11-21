"""Project a shared wave-1 snapshot onto Plan R slots for offline partial analysis."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from features import read_rows
from protocol import fingerprint, manifest
from submit import write_csv


def project(answers, prompts):
    by_id = {row["id"]: row["prompt"] for row in prompts}
    slots = manifest()
    aliases = {"F1": "flag-01", "F2": "flag-02", "F3": "flag-03"}
    aliases.update({f"L{s.id[-2:]}": s.id for s in slots if s.kind == "long"})
    yield {"event": "manifest", "sha256": fingerprint(slots), "source": "offline-wave1-projection",
           "scored": False}
    seen = set()
    for row in answers:
        source_id = row["id"]
        key = row["endpoint"], source_id
        if (source_id not in by_id or row["prompt"] != by_id[source_id]
                or row["repeat"] != 0 or key in seen):
            raise ValueError(f"unexpected or duplicate wave-1 record: {key}")
        seen.add(key)
        yield dict(row, event="answer", id=aliases.get(source_id, source_id), source_id=source_id)


def exact_flag_labels(rows, labels):
    flags = defaultdict(dict)
    for row in rows:
        source_id = row.get("source_id")
        if source_id in {"F3", "F4"} and (row.get("response") or "").strip():
            flags[row["endpoint"]][source_id] = row["response"]
    buckets = defaultdict(list)
    for ep, (family, _) in sorted(labels.items()):
        if flags[ep].keys() == {"F3", "F4"}:
            buckets[family, flags[ep]["F3"], flags[ep]["F4"]].append(ep)
    result = dict(labels)
    for (family, _, _), group in buckets.items():
        if len(group) > 1:
            matches = {result[ep][1] for ep in group}
            for ep, (f, model) in result.items():
                if f == family and model in matches:
                    result[ep] = (family, min(matches))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    journal = commands.add_parser("journal")
    journal.add_argument("answers", type=Path)
    journal.add_argument("prompts", type=Path)
    journal.add_argument("output", type=Path)
    experiment = commands.add_parser("exact-flags")
    experiment.add_argument("journal", type=Path)
    experiment.add_argument("baseline", type=Path)
    experiment.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    if args.command == "journal":
        try:
            rows = list(project(read_rows(args.answers), read_rows(args.prompts)))
        except ValueError as err:
            parser.error(str(err))
        with args.output.open("x", encoding="utf-8") as out:
            out.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    else:
        with args.baseline.open(encoding="utf-8", newline="") as handle:
            labels = {r["endpoint"]: (r["family"], r["model"]) for r in csv.DictReader(handle)}
        write_csv(args.output, exact_flag_labels(read_rows(args.journal), labels))


if __name__ == "__main__":
    main()
