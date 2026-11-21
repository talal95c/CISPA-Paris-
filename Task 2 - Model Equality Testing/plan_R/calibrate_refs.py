"""Fit optional solver calibration from previously collected reference journals."""
import argparse
import csv
import json
from pathlib import Path

from features import read_rows, signatures
from protocol import fingerprint, manifest
from replay import train
from simulate import VARIANTS


def fit_refs(journals, official_csv, size, phase=None):
    with official_csv.open(encoding="utf-8-sig", newline="") as f:
        official = {row["endpoint"]: (row["family"], row["model"]) for row in csv.DictReader(f)}
    endpoints = [f"ref-{i+1:02d}" for i in range(len(VARIANTS))]
    if set(official) != set(endpoints):
        raise ValueError("unexpected practice endpoints in truth file")
    truth = {ep: (official[ep], variant) for ep, variant in zip(endpoints, VARIANTS)}
    slots = manifest(size=size)
    training = []
    for i, path in enumerate(journals):
        rows = read_rows(path)
        if not rows or rows[0].get("scored") is not False or rows[0].get("sha256") != fingerprint(slots):
            raise ValueError(f"{path}: not a matching reference collector journal")
        sigs = signatures(rows, slots)
        if set(sigs) != set(endpoints):
            raise ValueError(f"{path}: missing or unexpected practice endpoints")
        training.append((i, sigs, truth))
    params, settings, flag_rates = train(training, phase or {})
    return {"params": params, "settings": settings, "flag_rates": flag_rates, "reference_journals": len(journals)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("journals", type=Path, nargs="+")
    p.add_argument("--practice-answers", type=Path, required=True)
    p.add_argument("--phase", type=Path, help="optional Phase 0 metrics from real references")
    p.add_argument("--items", type=int, choices=[12, *range(20, 26)], default=20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = fit_refs(args.journals, args.practice_answers, args.items,
                    json.loads(args.phase.read_text()) if args.phase else None)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{data['reference_journals']} reference journal(s): {args.output}")


if __name__ == "__main__":
    main()
