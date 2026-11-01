# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Streaming comparison of authoritative original-point outputs; no inference or GPU use."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path

SCHEMA = "synrex.runtime_comparison.v1"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_RECORD_BYTES = 65536
MODES = {"REFERENCE", "HYBRID", "NATIVE"}
SCOPES = {
    "whole_tile", "causal_streaming", "offline_lookahead", "file_export",
    "replay_looped", "replay_distinct", "loopback", "network_transport", "physical_sensor",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def sha256(value):
    require(isinstance(value, str) and len(value) == 64, "expected lowercase SHA256")
    require(all(c in "0123456789abcdef" for c in value), "expected lowercase SHA256")


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    def real(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result

    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=real)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("invalid JSON encoding or nesting") from error


def load_spec(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    require(len(raw) <= MAX_JSON_BYTES, "comparison spec exceeds byte limit")
    return strict_json(raw)


def metric_summary(confusion, classes):
    """Rows truth, columns prediction; absent unions are null, all-class means use zero."""
    count = len(classes)
    support = [sum(row) for row in confusion]
    predicted = [sum(row[c] for row in confusion) for c in range(count)]
    rows = []
    for c, definition in enumerate(classes):
        tp = confusion[c][c]
        union = support[c] + predicted[c] - tp
        rows.append({
            **definition, "support": support[c], "predicted": predicted[c],
            "iou": tp / union if union else None,
            "precision": tp / predicted[c] if predicted[c] else None,
            "recall": tp / support[c] if support[c] else None,
            "f1": 2 * tp / (support[c] + predicted[c])
            if support[c] + predicted[c] else None,
        })
    total = sum(support)
    present = [row["iou"] for row in rows if row["iou"] is not None]
    return {
        "confusion": confusion, "classes": rows, "scored_points": total,
        "accuracy": sum(confusion[c][c] for c in range(count)) / total if total else None,
        "mean_iou_present": sum(present) / len(present) if present else None,
        "mean_iou_all_classes": sum(row["iou"] or 0 for row in rows) / count
        if total else None,
        "macro_f1_all_classes": sum(row["f1"] or 0 for row in rows) / count
        if total else None,
    }


def ground_proxy(confusion, classes, ground_id):
    if ground_id is None:
        return None
    index = next(i for i, row in enumerate(classes) if row["id"] == ground_id)
    tp = confusion[index][index]
    predicted = sum(row[index] for row in confusion)
    support = sum(confusion[index])
    return {
        "status": "SEMANTIC_LABEL_PROXY", "ground_class_id": ground_id,
        "precision": tp / predicted if predicted else None,
        "recall": tp / support if support else None,
        "false_ground_points": predicted - tp,
        "false_ground_contamination": (predicted - tp) / predicted if predicted else None,
        "reference_ground_points": support,
        "survey_reference_available": False,
        "dtm_status": "BLOCKED_INPUT", "dtm_bias": None, "dtm_rmse": None,
        "dtm_quantiles": None, "ditch_breakline_retention": None,
    }


def validate_spec(spec):
    require(isinstance(spec, dict) and spec.get("schema") == SCHEMA, "unsupported schema")
    require(spec.get("track") in {"runtime_parity", "model_policy_quality"}, "invalid track")
    taxonomy = spec.get("taxonomy")
    require(isinstance(taxonomy, dict), "taxonomy required")
    classes = taxonomy.get("classes")
    require(isinstance(classes, list) and 1 <= len(classes) <= 256, "taxonomy size must be 1..256")
    ids = []
    names = []
    for row in classes:
        require(isinstance(row, dict) and set(row) == {"id", "name"}, "invalid class")
        require(integer(row["id"]) and 0 <= row["id"] <= 255, "invalid class id")
        require(nonempty(row["name"]), "class name required")
        ids.append(row["id"])
        names.append(row["name"])
    require(len(set(ids)) == len(ids) and len(set(names)) == len(names), "duplicate class")
    ignored = taxonomy.get("ignored_reference_labels")
    require(isinstance(ignored, list) and all(integer(v) for v in ignored),
            "invalid ignored labels")
    require(len(set(ignored)) == len(ignored) and not set(ids) & set(ignored),
            "ignored label overlap")
    ground = taxonomy.get("ground_class_id")
    require(ground is None or integer(ground) and ground in ids, "invalid ground class")
    sources = spec.get("sources")
    require(isinstance(sources, list) and sources, "nonempty authoritative sources required")
    hashes = []
    for source in sources:
        require(isinstance(source, dict) and set(source) == {"sha256", "points"}, "invalid source")
        sha256(source["sha256"])
        require(integer(source["points"]) and source["points"] >= 0, "invalid source point count")
        hashes.append(source["sha256"])
    require(len(set(hashes)) == len(hashes), "duplicate source identity")
    require(sum(row["points"] for row in sources) > 0, "no authoritative points")
    evaluation = spec.get("evaluation")
    require(isinstance(evaluation, dict) and nonempty(evaluation.get("split")), "split required")
    require(nonempty(evaluation.get("protocol")), "evaluation protocol required")
    require("latency_budget_ms" in evaluation, "declare latency budget or null")
    validate_number(evaluation["latency_budget_ms"], "latency budget")
    for name in ("reference", "candidate"):
        validate_run(spec.get(name), canonical_hash(taxonomy), canonical_hash(sources))
    if spec["track"] == "runtime_parity":
        a, b = spec["reference"], spec["candidate"]
        require(a["provenance"]["model_sha256"] == b["provenance"]["model_sha256"],
                "parity model_sha256 mismatch")
        require(a["provenance"]["package_taxonomy_sha256"]
                == b["provenance"]["package_taxonomy_sha256"],
                "parity package taxonomy mismatch")
        require(canonical_hash(a["semantics"]) == canonical_hash(b["semantics"]),
                "parity semantics/windows/preprocessing mismatch")
        threshold = spec.get("minimum_label_agreement")
        require(type(threshold) in (float, int) and 0 <= threshold <= 1,
                "preregister minimum_label_agreement in [0,1]")
    else:
        require("minimum_label_agreement" not in spec, "policy quality cannot use parity gate")
    return classes, ignored, ground


def validate_number(value, name):
    require(value is None or type(value) in (int, float) and value >= 0
            and (integer(value) or math.isfinite(value)),
            f"invalid {name}; unavailable values must be null")


def validate_run(run, taxonomy_hash, data_hash):
    require(isinstance(run, dict) and nonempty(run.get("name")), "named run required")
    require(run.get("execution_mode") in MODES, "explicit execution mode required")
    require(run.get("execution_scope") in SCOPES, "explicit execution scope required")
    require(isinstance(run.get("limitations"), list)
            and all(nonempty(v) for v in run["limitations"]), "limitations list required")
    unavailable = run.get("unsupported_modes")
    require(isinstance(unavailable, dict)
            and all(k in MODES and nonempty(v) for k, v in unavailable.items()),
            "unsupported_modes must map modes to reasons")
    require(run["execution_mode"] not in unavailable, "executed mode marked unsupported")
    provenance = run.get("provenance")
    require(isinstance(provenance, dict), "provenance required")
    for field in ("source_sha256", "config_sha256", "model_sha256", "source_manifest_sha256",
                  "scoring_taxonomy_sha256"):
        sha256(provenance.get(field))
    require("engine_sha256" in provenance, "declare engine identity or null")
    if provenance["engine_sha256"] is not None:
        sha256(provenance["engine_sha256"])
    require(provenance["scoring_taxonomy_sha256"] == taxonomy_hash, "taxonomy identity mismatch")
    require(provenance["source_manifest_sha256"] == data_hash, "data identity mismatch")
    require("package_taxonomy_sha256" in provenance
            and "package_taxonomy_unavailable_reason" in provenance,
            "declare package taxonomy identity and availability reason")
    package_hash = provenance["package_taxonomy_sha256"]
    reason = provenance["package_taxonomy_unavailable_reason"]
    if package_hash is None:
        require(nonempty(reason), "unavailable package taxonomy requires explicit reason")
    else:
        sha256(package_hash)
        require(reason is None, "available package taxonomy reason must be null")
    require(nonempty(provenance.get("hardware")) and nonempty(provenance.get("software")),
            "hardware/software required")
    semantics = run.get("semantics")
    require(isinstance(semantics, dict), "semantics required")
    fields = semantics.get("input_fields")
    require(isinstance(fields, list) and fields and all(nonempty(v) for v in fields)
            and len(set(fields)) == len(fields), "input fields required")
    for field in ("preprocessing", "windows", "reconstruction"):
        require(isinstance(semantics.get(field), dict) and semantics[field], f"declare {field}")
    for field in ("precision", "policy", "architecture", "input_provenance"):
        require(nonempty(semantics.get(field)), f"declare {field}")
    timing = run.get("measurements")
    require(isinstance(timing, dict) and nonempty(timing.get("timing_boundary")),
            "timing boundary required")
    for field in ("offered_points_per_second", "achieved_points_per_second", "p50_latency_ms",
                  "p95_latency_ms", "p99_latency_ms", "max_latency_ms", "peak_memory_bytes",
                  "energy_joules"):
        require(field in timing, f"declare {field} or null")
        validate_number(timing[field], field)
    require(timing["peak_memory_bytes"] is None or integer(timing["peak_memory_bytes"]),
            "peak_memory_bytes must be integer or null")
    # Normalize/validate every extra metadata value too, including API calls with nonfinite values.
    canonical_hash(run)


class Records:
    def __init__(self, stack, path):
        self.stream = stack.enter_context(Path(path).open("rb"))
        self.digest = hashlib.sha256()
        self.line = 0

    def next(self):
        raw = self.stream.readline(MAX_RECORD_BYTES + 1)
        require(len(raw) <= MAX_RECORD_BYTES, "record exceeds byte limit")
        if not raw:
            return None
        self.digest.update(raw)
        self.line += 1
        row = strict_json(raw)
        require(isinstance(row, dict) and set(row) == {"source_sha256", "point_index", "label"},
                f"invalid record fields at line {self.line}")
        require(integer(row["point_index"]) and row["point_index"] >= 0,
                f"invalid point index at line {self.line}")
        require(integer(row["label"]), f"invalid label at line {self.line}")
        return row


def compare(spec, truth_path, reference_path, candidate_path):
    """Score canonical source/index streams with O(K² + source manifest) memory.

    The supplied frozen source manifest is authoritative; this function does not reread
    LAS files or infer source identities from arrival order. Adapters must sort outputs
    into manifest source order and original point index before calling this function.
    """
    # Detach caller objects and reject non-JSON/nonfinite metadata before processing files.
    spec = strict_json(json.dumps(spec, allow_nan=False))
    classes, ignored, ground = validate_spec(spec)
    class_index = {row["id"]: i for i, row in enumerate(classes)}
    allowed_truth = set(class_index) | set(ignored)
    count = len(classes)
    confusion = {name: [[0] * count for _ in range(count)]
                 for name in ("reference", "candidate")}
    disagreement = [[0] * count for _ in range(count)]
    points = ignored_points = differing = 0
    with ExitStack() as stack:
        readers = {name: Records(stack, path) for name, path in (
            ("truth", truth_path), ("reference", reference_path), ("candidate", candidate_path))}
        for source in spec["sources"]:
            for index in range(source["points"]):
                rows = {}
                for name, reader in readers.items():
                    row = reader.next()
                    require(row is not None, f"{name}: missing original point")
                    require(row["source_sha256"] == source["sha256"]
                            and row["point_index"] == index,
                            f"{name}: source/index mismatch, duplicate, missing or reordered point")
                    rows[name] = row["label"]
                    allowed = allowed_truth if name == "truth" else class_index
                    require(row["label"] in allowed, f"{name}: label outside taxonomy")
                points += 1
                a, b = class_index[rows["reference"]], class_index[rows["candidate"]]
                disagreement[a][b] += 1
                differing += a != b
                if rows["truth"] in ignored:
                    ignored_points += 1
                    continue
                truth = class_index[rows["truth"]]
                confusion["reference"][truth][a] += 1
                confusion["candidate"][truth][b] += 1
        for name, reader in readers.items():
            require(reader.next() is None, f"{name}: extra or duplicate original point")
        input_hashes = {name: reader.digest.hexdigest() for name, reader in readers.items()}
    scores = {name: metric_summary(matrix, classes) for name, matrix in confusion.items()}
    agreement = (points - differing) / points
    parity = None
    if spec["track"] == "runtime_parity":
        threshold = spec["minimum_label_agreement"]
        parity = {
            "label_agreement": agreement, "minimum_label_agreement": threshold,
            "label_agreement_verdict": "PASS" if agreement >= threshold else "FAIL",
            "scope": "Original-point labels only; preprocessing/logit qualification is external.",
            "package_taxonomy_binding": "MATCHED_DECLARED_ARTIFACT_HASH"
            if spec["reference"]["provenance"]["package_taxonomy_sha256"] else "UNAVAILABLE",
            "package_taxonomy_qualification": "NOT_ASSESSED",
        }
    deltas = {}
    for metric in ("accuracy", "mean_iou_all_classes", "macro_f1_all_classes"):
        a, b = scores["reference"][metric], scores["candidate"][metric]
        deltas[metric] = b - a if a is not None and b is not None else None
    return {
        "schema": SCHEMA, "measurement_status": "COMPLETE", "track": spec["track"],
        "qualification_status": "NOT_ASSESSED", "spec": spec,
        "spec_sha256": canonical_hash(spec), "input_record_sha256": input_hashes,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coverage": {"original_points": points, "reference_points": points,
                     "candidate_points": points, "scored_points": points - ignored_points,
                     "ignored_points": ignored_points, "missing_points": 0},
        "metric_rules": {
            "confusion": "rows=truth, columns=prediction; class order follows taxonomy",
            "ignored": "excluded from quality metrics; required in coverage and label agreement",
            "undefined": "zero denominator -> null; absent union -> null",
            "all_class_macro": "absent classes contribute zero; no scored points -> null",
            "statistical_unit": (
                "No confidence/quality verdict; scene/acquisition analysis external"
            ),
        },
        "quality": scores, "candidate_minus_reference": deltas,
        "label_disagreement": {"points": differing, "agreement": agreement,
                               "confusion_reference_candidate": disagreement},
        "numerical_parity": parity,
        "ground_proxy": {name: ground_proxy(matrix, classes, ground)
                         for name, matrix in confusion.items()},
        "limitations": ["Metadata declarations do not verify preprocessing, causality or timings.",
                        "No inference, model quality qualification or survey terrain measurement.",
                        "Authoritative source hashes/counts supplied by frozen input manifest."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "truth", "reference", "candidate", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(args.out.resolve() not in {p.resolve() for p in (
        args.spec, args.truth, args.reference, args.candidate)}, "output aliases an input")
    report = compare(load_spec(args.spec), args.truth, args.reference, args.candidate)
    # Exclusive creation prevents replacing an earlier receipt or a source via hard link.
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"measurement_status": report["measurement_status"],
                      "numerical_parity": report["numerical_parity"]}))
    return 1 if report["numerical_parity"] and (
        report["numerical_parity"]["label_agreement_verdict"] == "FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
