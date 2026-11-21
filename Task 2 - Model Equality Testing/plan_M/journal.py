"""Normalize shared collector and four-field response journals."""

import json
from pathlib import Path
import sys


def load_shared(path: Path):
    """Load the collector protocol and analysis primitives from the Plan R checkout."""
    root = path.resolve()
    for name in ("protocol", "features", "partition", "submit"):
        if not (root / f"{name}.py").is_file():
            raise ValueError(f"Missing Plan R module {name}.py under {root}; set --shared-dir")
        prior = sys.modules.get(name)
        if prior is not None and Path(prior.__file__).parent != root:
            raise ValueError(f"Cannot load a second common implementation for {name}")
    sys.path.insert(0, str(root))
    import features
    import partition
    import protocol
    import submit
    return features, partition, protocol, submit


def manifest_from_json(path, protocol):
    return [
        protocol.Slot(**{**row, "items": tuple(protocol.Item(**item) for item in row["items"])})
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]


def read_jsonl(path: Path, *, allow_incomplete_tail=False):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if allow_incomplete_tail and not line.endswith("\n"):
                    return
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield row


def normalize_rows(rows, slots, *, allow_partial=False, expected=None):
    """Return scored text rows, preserving committed empty and rejected slots."""
    manifest = {slot.id: slot for slot in slots}
    if len(manifest) != len(slots):
        raise ValueError("The manifest contains duplicate slot IDs")
    by_prompt = {}
    for slot in slots:
        by_prompt.setdefault(slot.prompt, []).append(slot.id)
    committed = {}
    for entry in rows:
        event = entry.get("event")
        if event not in (None, "answer", "rejected"):
            continue
        if event is None and not {"endpoint_id", "prompt_id", "prompt", "response"} <= entry.keys():
            raise ValueError("Text rows require endpoint_id, prompt_id, prompt and response")
        endpoint = entry.get("endpoint_id", entry.get("endpoint"))
        slot_id = entry.get("prompt_id", entry.get("id"))
        probe_id = entry.get("probe_id", slot_id)
        prompt = entry.get("prompt")
        response = entry.get("response")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("Every response must name an endpoint")
        if not isinstance(slot_id, str) or slot_id not in manifest:
            raise ValueError(f"{endpoint}: unknown prompt_id {slot_id!r}")
        if not isinstance(prompt, str):
            raise ValueError(f"{endpoint}/{slot_id}: expected prompt text")
        inferred_repair = slot_id.startswith("reserve-") and "probe_id" not in entry and prompt != manifest[slot_id].prompt
        if inferred_repair:
            probe_id = None
        elif not isinstance(probe_id, str) or probe_id not in manifest:
            raise ValueError(f"{endpoint}/{slot_id}: unknown probe_id {probe_id!r}")
        elif slot_id != probe_id and not slot_id.startswith("reserve-"):
            raise ValueError(f"{endpoint}/{slot_id}: only reserves may repair another probe")
        elif prompt != manifest[probe_id].prompt:
            raise ValueError(f"{endpoint}/{slot_id}: prompt does not match {probe_id} in the manifest")
        if not isinstance(response, (str, type(None))):
            raise ValueError(f"{endpoint}/{slot_id}: expected response text or null")
        key = endpoint, slot_id
        row = dict(event=event or "answer", endpoint=endpoint, id=slot_id,
                   probe_id=probe_id, prompt=prompt, response=response or "")
        if key in committed and committed[key] != row:
            raise ValueError(f"{endpoint}/{slot_id}: conflicting completed responses")
        committed[key] = row
    repaired = {(ep, row["probe_id"]) for (ep, name), row in committed.items()
                if row["probe_id"] not in (None, name)}
    for (ep, name), row in sorted(committed.items()):
        if row["probe_id"] is not None:
            continue
        matches = [pid for pid in by_prompt.get(row["prompt"], []) if not pid.startswith("reserve-")]
        if not matches:
            raise ValueError(f"{ep}/{name}: reserve prompt is outside the manifest")
        missing = [pid for pid in matches if (ep, pid) not in repaired and
                   not committed.get((ep, pid), {}).get("response", "").strip()]
        row["probe_id"] = (missing or matches)[0]
        repaired.add((ep, row["probe_id"]))
    endpoints = sorted({key[0] for key in committed})
    if not endpoints:
        raise ValueError("The journal contains no completed slots")
    if expected is not None and set(endpoints) != set(expected):
        raise ValueError("The journal endpoint set does not match the target")
    if not allow_partial:
        missing = sum((endpoint, name) not in committed for endpoint in endpoints for name in manifest)
        if missing:
            raise ValueError(f"{missing} unfinished slots; use --allow-partial for a provisional CSV")
    # Keep original observations ahead of any reserves that reuse their prompts.
    normalized = sorted(committed.values(), key=lambda row: row["id"].startswith("reserve-"))
    return normalized, endpoints


def long_answers(rows, slots):
    """Join reserve repairs by their effective prompt ID and ignore empty text."""
    manifest = {slot.id: slot for slot in slots if slot.kind == "long"}
    answers = {}
    for row in rows:
        probe = row["probe_id"]
        if probe in manifest and row["event"] == "answer" and row["response"].strip():
            texts = answers.setdefault(row["endpoint"], {})
            if probe == row["id"] or probe not in texts:
                texts[probe] = row["response"]
    return answers, manifest
