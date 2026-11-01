# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Identity-bound Q0/Q1 validation evaluation; no inference or GPU scheduling.

Traces are checked evidence, not proof that a producer really executed its declaration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

_module = importlib.util.spec_from_file_location(
    "_quality_comparison", Path(__file__).with_name("runtime_comparison.py")
)
comparison = importlib.util.module_from_spec(_module)
_module.loader.exec_module(comparison)
require = comparison.require
integer = comparison.integer
canonical_hash = comparison.canonical_hash
SCHEMA = "synrex.runtime_quality.v1"
Q1_POINTS = 16384
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_RECORDS = 50_000
MAX_PARSED_BYTES = 64 * 1024 * 1024
MAX_TOTAL_ARTIFACT_BYTES = 1024 * 1024 * 1024
MAX_MATRIX_CELLS = 1_000_000
MAX_SOURCES = 64
MAX_NS = (1 << 64) - 1
RUNS = ("q0", "q1_reference", "q1_native")
BINDINGS = ("model", "preprocessing", "precision", "source_checkout")


def timestamp(value):
    return integer(value) and 0 <= value <= MAX_NS


def digest_file(path, byte_budget=None):
    digest = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if byte_budget is not None:
                byte_budget[0] -= len(chunk)
                require(byte_budget[0] >= 0, "aggregate hash byte limit exceeded")
            require(total <= MAX_ARTIFACT_BYTES, "artifact exceeds bounded validation subset limit")
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(base, entry):
    require(isinstance(entry, dict) and set(entry) <= {"path", "sha256"}, "invalid artifact entry")
    require(isinstance(entry.get("path"), str) and entry["path"], "artifact path required")
    path = Path(entry["path"])
    require(
        not path.is_absolute() and ".." not in path.parts,
        "artifact path must be relative without ..",
    )
    base = Path(base).resolve()
    path = base / path
    require(path.resolve().is_relative_to(base), "artifact escapes base directory")
    return path


def _admit_artifact_work(spec, base_dir, include_outputs):
    """Bound map cardinality, file types and total I/O before the first hash read."""
    require(isinstance(spec, dict), "quality spec object required")
    sources = spec.get("sources")
    require(
        isinstance(sources, list) and 1 <= len(sources) <= MAX_SOURCES,
        "bounded authoritative sources required",
    )
    artifacts = spec.get("artifacts")
    require(
        isinstance(artifacts, dict) and len(artifacts) <= MAX_SOURCES + 12,
        "bounded artifact map required",
    )
    total = 0
    for name, entry in artifacts.items():
        path = artifact_path(base_dir, entry)
        if not include_outputs and any(
            name == f"{run}_{kind}" for run in RUNS for kind in ("predictions", "windows")
        ):
            continue
        require(path.is_file(), "regular artifact file required")
        size = path.stat().st_size
        require(size <= MAX_ARTIFACT_BYTES, "artifact exceeds bounded validation subset limit")
        total += size
        require(total <= MAX_TOTAL_ARTIFACT_BYTES, "aggregate artifact byte limit exceeded")


def freeze_protocol(spec, base_dir):
    """Return input/protocol identity before outputs exist; retain it before inference.

    Output paths and contents are deliberately excluded. This cannot attest when an
    external caller actually preregistered the identity.
    """
    _admit_artifact_work(spec, base_dir, include_outputs=False)
    byte_budget = [MAX_TOTAL_ARTIFACT_BYTES]
    frozen = comparison.strict_json(json.dumps(spec, allow_nan=False))
    frozen.pop("protocol_sha256", None)
    frozen.pop("manifest_sha256", None)
    for name in list(frozen["artifacts"]):
        if any(name == f"{run}_{kind}" for run in RUNS for kind in ("predictions", "windows")):
            del frozen["artifacts"][name]
        else:
            entry = frozen["artifacts"][name]
            digest = digest_file(artifact_path(base_dir, entry), byte_budget)
            require(
                "sha256" not in entry or entry["sha256"] == digest,
                "existing input identity mismatch",
            )
            frozen["artifacts"][name] = {"sha256": digest}
    return canonical_hash(frozen)


def seal_manifest(spec, base_dir):
    """Hash actual evaluation artifacts after inference against a preregistered protocol.

    Existing bindings are never silently refreshed. Write the result with exclusive
    creation and retain its manifest_sha256 outside the mutable manifest file.
    """
    _admit_artifact_work(spec, base_dir, include_outputs=True)
    byte_budget = [MAX_TOTAL_ARTIFACT_BYTES]
    spec = comparison.strict_json(json.dumps(spec, allow_nan=False))
    require("manifest_sha256" not in spec, "manifest already sealed")
    require(
        spec.get("protocol_sha256") == freeze_protocol(spec, base_dir),
        "preregistered protocol identity mismatch",
    )
    require(isinstance(spec.get("artifacts"), dict), "artifact map required")
    for entry in spec["artifacts"].values():
        digest = digest_file(artifact_path(base_dir, entry), byte_budget)
        require(
            "sha256" not in entry or entry["sha256"] == digest,
            "existing artifact identity mismatch",
        )
        entry["sha256"] = digest
    _validate_manifest(spec)
    spec["manifest_sha256"] = canonical_hash(spec)
    return spec


def _validate_manifest(spec):
    require(spec.get("schema") == SCHEMA, "unsupported quality schema")
    comparison.sha256(spec.get("protocol_sha256"))
    require(
        spec.get("purpose") == "validation_development", "only validation development is supported"
    )
    require(
        spec.get("split") == "validation", "consumed test/holdout data cannot select policy here"
    )
    require(
        spec.get("clock") == "shared_monotonic_ns", "one shared monotonic nanosecond clock required"
    )
    require(
        spec.get("policy")
        == {"q0": "whole_source_minimum_v1", "q1": "independent_minimum_16384_eof_partial_v1"},
        "frozen Q0/Q1 policy required",
    )
    require(
        spec.get("gates")
        == {
            "minimum_label_agreement": 0.995,
            "maximum_mean_iou_loss": 0.01,
            "maximum_supported_class_iou_loss": 0.02,
            "supported_class_points": 1000,
        },
        "v1 numerical gates must be frozen before measurement",
    )
    taxonomy = spec.get("taxonomy")
    require(
        isinstance(taxonomy, dict) and set(taxonomy) == {"classes", "ignored_reference_labels"},
        "explicit taxonomy and ignored labels required",
    )
    classes, ignored = taxonomy["classes"], taxonomy["ignored_reference_labels"]
    require(isinstance(classes, list) and 1 <= len(classes) <= 256, "invalid taxonomy")
    for row in classes:
        require(
            isinstance(row, dict)
            and set(row) == {"id", "name"}
            and integer(row["id"])
            and 0 <= row["id"] <= 255
            and comparison.nonempty(row["name"]),
            "invalid class",
        )
    ids = [row["id"] for row in classes]
    names = [row["name"] for row in classes]
    require(
        len(set(ids)) == len(ids) and len(set(names)) == len(names), "duplicate class id or name"
    )
    require(
        isinstance(ignored, list)
        and all(integer(v) for v in ignored)
        and len(set(ignored)) == len(ignored)
        and not set(ids) & set(ignored),
        "invalid ignored labels",
    )
    sources = spec.get("sources")
    require(
        isinstance(sources, list) and 1 <= len(sources) <= MAX_SOURCES,
        "bounded authoritative sources required",
    )
    artifacts = spec["artifacts"]
    required = {*BINDINGS, "observations", "truth"}
    required.update(f"{run}_{kind}" for run in RUNS for kind in ("predictions", "windows"))
    reserved = set(required)
    for source in sources:
        require(
            isinstance(source, dict)
            and set(source) == {"artifact", "points", "scene", "acquisition"},
            "invalid source metadata",
        )
        require(
            isinstance(source["artifact"], str)
            and integer(source["points"])
            and source["points"] > 0
            and comparison.nonempty(source["scene"])
            and comparison.nonempty(source["acquisition"]),
            "invalid authoritative source",
        )
        require(
            source["artifact"] not in reserved,
            "source artifact collides with reserved input/output",
        )
        required.add(source["artifact"])
    require(
        len({source["artifact"] for source in sources}) == len(sources), "duplicate source artifact"
    )
    require(required == set(artifacts), "missing or unused artifact binding")
    require(
        (1 + len({source["acquisition"] for source in sources})) * len(RUNS) * len(classes) ** 2
        <= MAX_MATRIX_CELLS,
        "aggregate acquisition matrix work limit exceeded",
    )
    for entry in artifacts.values():
        require(isinstance(entry, dict) and set(entry) == {"path", "sha256"}, "unsealed artifact")
        comparison.sha256(entry["sha256"])
    identities = [artifacts[source["artifact"]]["sha256"] for source in sources]
    require(len(set(identities)) == len(identities), "duplicate source identity")
    require(
        sum(source["points"] for source in sources) <= MAX_RECORDS,
        "validation subset point limit exceeded",
    )


def _records(path, expected_sha256=None, byte_budget=None):
    rows = []
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while raw := stream.readline(comparison.MAX_RECORD_BYTES + 1):
            require(len(raw) <= comparison.MAX_RECORD_BYTES, "record exceeds byte limit")
            digest.update(raw)
            if byte_budget is not None:
                byte_budget[0] -= len(raw)
                require(byte_budget[0] >= 0, "aggregate parsed byte limit exceeded")
            require(len(rows) < MAX_RECORDS, "record count limit exceeded")
            row = comparison.strict_json(raw)
            require(isinstance(row, dict), "record must be object")
            rows.append(row)
    require(
        expected_sha256 is None or digest.hexdigest() == expected_sha256,
        "consumed record artifact identity mismatch",
    )
    return rows


def _decimal(value):
    require(isinstance(value, str), "coordinates must retain decimal source precision as strings")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid decimal coordinate") from error
    require(
        result.is_finite() and len(value) <= 128 and abs(result.adjusted()) <= 100,
        "nonfinite or unbounded decimal coordinate",
    )
    return result


def _decimal_string(value):
    return (
        format(value, "f").rstrip("0").rstrip(".")
        if value and value.as_tuple().exponent < 0
        else format(value, "f")
        if value
        else "0"
    )


def _identity(row):
    return row.get("source_sha256"), row.get("point_index")


def validate_observations(rows, sources):
    require(
        len(rows) == sum(source["points"] for source in sources), "observation coverage mismatch"
    )
    position = 0
    for source in sources:
        previous = -1
        for index in range(source["points"]):
            row = rows[position]
            require(
                set(row) == {"source_sha256", "point_index", "available_ns", "captured_ns", "xyz"},
                "observations must contain feature geometry and timing only; labels forbidden",
            )
            require(
                _identity(row) == (source["sha256"], index) and integer(row["point_index"]),
                "observation source/index mismatch",
            )
            require(
                timestamp(row["available_ns"]) and row["available_ns"] >= previous,
                "nonmonotonic or invalid availability",
            )
            require(
                row["captured_ns"] is None
                or timestamp(row["captured_ns"])
                and row["captured_ns"] <= row["available_ns"],
                "invalid capture clock",
            )
            require(isinstance(row["xyz"], list) and len(row["xyz"]) == 3, "xyz required")
            for coordinate in row["xyz"]:
                _decimal(coordinate)
            previous = row["available_ns"]
            position += 1


def build_reference_windows(rows, policy):
    """Independent contiguous-source oracle. Q1 has no timeout flush or context.

    Input must first pass validate_observations. Window input digests exclude labels
    by construction; source identity, timing and exact decimal geometry are bound.
    """
    require(policy in {"q0", "q1"}, "unknown reference policy")
    result = []
    start = 0
    while start < len(rows):
        source = rows[start]["source_sha256"]
        end = start + 1
        while end < len(rows) and rows[end]["source_sha256"] == source:
            end += 1
        size = end - start if policy == "q0" else Q1_POINTS
        window_index = 0
        for begin in range(start, end, size):
            chunk = rows[begin : min(begin + size, end)]
            result.append(
                {
                    "source_sha256": source,
                    "window_index": window_index,
                    "begin": chunk[0]["point_index"],
                    "end": chunk[-1]["point_index"] + 1,
                    "cutoff_ns": chunk[-1]["available_ns"],
                    "origin": [
                        _decimal_string(min(_decimal(row["xyz"][axis]) for row in chunk))
                        for axis in range(3)
                    ],
                    "input_sha256": canonical_hash(chunk),
                }
            )
            window_index += 1
        start = end
    return result


def cutoff_reference(rows, cutoff_ns):
    """Offline diagnostic context containing exactly observations available by cutoff.

    This does not call a native window builder and does not treat the full-file
    minimum as an online anchor. Equal availability timestamps are included together.
    """
    require(timestamp(cutoff_ns), "invalid cutoff")
    available = [row for row in rows if row["available_ns"] <= cutoff_ns]
    return {
        "cutoff_ns": cutoff_ns,
        "observations": available,
        "input_sha256": canonical_hash(available),
        "whole_available_reference_windows": build_reference_windows(available, "q0"),
    }


def _check_windows(actual, expected, bindings):
    require(len(actual) == len(expected), "window count mismatch")
    for row, oracle in zip(actual, expected, strict=True):
        require(
            set(row) == set(oracle) | {"bindings", "admitted_ns", "output_ns"},
            "invalid window trace fields",
        )
        require(
            all(integer(row[key]) for key in ("window_index", "begin", "end", "cutoff_ns")),
            "invalid discrete window trace",
        )
        require(
            {key: row[key] for key in oracle} == oracle,
            "window membership/cutoff/origin/input identity mismatch",
        )
        require(
            row["bindings"] == bindings,
            "window model/preprocessing/precision/source identity mismatch",
        )
        require(
            timestamp(row["admitted_ns"])
            and timestamp(row["output_ns"])
            and row["cutoff_ns"] <= row["admitted_ns"] <= row["output_ns"],
            "invalid service clock",
        )


def _summary(values):
    if not values:
        return {"count": 0, "mean_ns": None, "p50_ns": None, "p99_ns": None, "max_ns": None}
    values = sorted(values)
    return {
        "count": len(values),
        "mean_ns": sum(values) / len(values),
        "p50_ns": values[math.ceil(len(values) * 0.5) - 1],
        "p99_ns": values[math.ceil(len(values) * 0.99) - 1],
        "max_ns": values[-1],
    }


def point_age(rows, windows):
    """Availability accumulation, admission queue and service are distinct boundaries."""
    lookup = {(w["source_sha256"], i): w for w in windows for i in range(w["begin"], w["end"])}
    ages = {
        key: []
        for key in (
            "availability_accumulation",
            "queue",
            "service",
            "availability_to_output",
            "capture_to_output",
        )
    }
    for row in rows:
        window = lookup[_identity(row)]
        ages["availability_accumulation"].append(window["cutoff_ns"] - row["available_ns"])
        ages["queue"].append(window["admitted_ns"] - window["cutoff_ns"])
        ages["service"].append(window["output_ns"] - window["admitted_ns"])
        ages["availability_to_output"].append(window["output_ns"] - row["available_ns"])
        if row["captured_ns"] is not None:
            ages["capture_to_output"].append(window["output_ns"] - row["captured_ns"])
    return {name: _summary(values) for name, values in ages.items()}


def _deltas(reference, candidate):
    result = {}
    for name in ("accuracy", "mean_iou_all_classes", "macro_f1_all_classes"):
        a, b = reference[name], candidate[name]
        result[name] = b - a if a is not None and b is not None else None
    result["classes"] = [
        {
            "id": a["id"],
            "support": a["support"],
            "iou_delta": b["iou"] - a["iou"]
            if a["iou"] is not None and b["iou"] is not None
            else None,
        }
        for a, b in zip(reference["classes"], candidate["classes"], strict=True)
    ]
    return result


def evaluate(manifest, base_dir, expected_manifest_sha256, expected_protocol_sha256):
    """Evaluate a preregistered identity against actual files; never consumes test tuning."""
    manifest = comparison.strict_json(json.dumps(manifest, allow_nan=False))
    identity = manifest.pop("manifest_sha256", None)
    comparison.sha256(expected_manifest_sha256)
    require(
        identity == expected_manifest_sha256 == canonical_hash(manifest),
        "frozen manifest identity mismatch",
    )
    _validate_manifest(manifest)
    require(
        manifest["protocol_sha256"]
        == expected_protocol_sha256
        == freeze_protocol(manifest, base_dir),
        "independently retained protocol identity mismatch",
    )
    artifacts = manifest["artifacts"]
    paths = {}
    total_artifact_bytes = 0
    for name, entry in artifacts.items():
        paths[name] = artifact_path(base_dir, entry)
        total_artifact_bytes += paths[name].stat().st_size
        require(
            total_artifact_bytes <= MAX_TOTAL_ARTIFACT_BYTES,
            "aggregate artifact byte limit exceeded",
        )
        require(digest_file(paths[name]) == entry["sha256"], f"artifact identity mismatch: {name}")
    sources = [
        {**source, "sha256": artifacts[source["artifact"]]["sha256"]}
        for source in manifest["sources"]
    ]
    byte_budget = [MAX_PARSED_BYTES]

    def read(name):
        return _records(paths[name], artifacts[name]["sha256"], byte_budget)

    rows = read("observations")
    validate_observations(rows, sources)
    bindings = {name + "_sha256": artifacts[name]["sha256"] for name in BINDINGS}
    expected = {
        "q0": build_reference_windows(rows, "q0"),
        "q1": build_reference_windows(rows, "q1"),
    }
    windows = {}
    predictions = {}
    for run in RUNS:
        windows[run] = read(f"{run}_windows")
        _check_windows(windows[run], expected["q0" if run == "q0" else "q1"], bindings)
        predictions[run] = read(f"{run}_predictions")
        require(len(predictions[run]) == len(rows), f"{run}: original-point coverage mismatch")
    truth = read("truth")
    require(len(truth) == len(rows), "truth coverage mismatch")
    classes = manifest["taxonomy"]["classes"]
    class_index = {row["id"]: index for index, row in enumerate(classes)}
    ignored = manifest["taxonomy"]["ignored_reference_labels"]

    def matrices():
        return {run: [[0] * len(classes) for _ in classes] for run in RUNS}

    confusion = matrices()
    acquisition = {source["acquisition"]: matrices() for source in sources}
    source_acquisition = {source["sha256"]: source["acquisition"] for source in sources}
    disagreements = {"q0_vs_q1_native": 0, "q1_reference_vs_native": 0}
    position = {
        run: {
            (w["source_sha256"], index): w["window_index"]
            for w in windows[run]
            for index in range(w["begin"], w["end"])
        }
        for run in RUNS
    }
    parity_confusion = [[0] * len(classes) for _ in classes]
    scored = 0
    for index, observation in enumerate(rows):
        label = truth[index]
        require(
            set(label) == {"source_sha256", "point_index", "label"}
            and _identity(label) == _identity(observation)
            and integer(label["point_index"]),
            "truth source/index mismatch",
        )
        require(
            integer(label["label"]) and label["label"] in set(class_index) | set(ignored),
            "invalid truth label",
        )
        labels = {}
        for run in RUNS:
            prediction = predictions[run][index]
            require(
                set(prediction) == {"source_sha256", "point_index", "window_index", "label"}
                and _identity(prediction) == _identity(observation)
                and integer(prediction["point_index"])
                and integer(prediction["window_index"])
                and prediction["window_index"] == position[run][_identity(observation)],
                f"{run}: source/index/window mismatch; revised/map outputs forbidden",
            )
            require(
                integer(prediction["label"]) and prediction["label"] in class_index,
                "invalid prediction label",
            )
            labels[run] = prediction["label"]
        parity_confusion[class_index[labels["q1_reference"]]][class_index[labels["q1_native"]]] += 1
        disagreements["q0_vs_q1_native"] += labels["q0"] != labels["q1_native"]
        disagreements["q1_reference_vs_native"] += labels["q1_reference"] != labels["q1_native"]
        if label["label"] in ignored:
            continue
        scored += 1
        a = class_index[label["label"]]
        for run in RUNS:
            b = class_index[labels[run]]
            confusion[run][a][b] += 1
            acquisition[source_acquisition[observation["source_sha256"]]][run][a][b] += 1
    quality = {run: comparison.metric_summary(matrix, classes) for run, matrix in confusion.items()}
    deltas = _deltas(quality["q1_reference"], quality["q1_native"])
    agreement = 1 - disagreements["q1_reference_vs_native"] / len(rows)
    gates = manifest["gates"]
    supported = [
        row for row in deltas["classes"] if row["support"] >= gates["supported_class_points"]
    ]
    label_gate = agreement >= gates["minimum_label_agreement"]
    quality_gate = (
        deltas["mean_iou_all_classes"] >= -gates["maximum_mean_iou_loss"]
        if deltas["mean_iou_all_classes"] is not None
        else None
    )
    class_gate = all(
        row["iou_delta"] is not None
        and row["iou_delta"] >= -gates["maximum_supported_class_iou_loss"]
        for row in supported
    )
    return {
        "schema": SCHEMA,
        "manifest_sha256": identity,
        "protocol_sha256": expected_protocol_sha256,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison_scorer_sha256": hashlib.sha256(
            Path(comparison.__file__).read_bytes()
        ).hexdigest(),
        "measurement_status": "ARTIFACT_EVALUATION_COMPLETE",
        "qualification_status": "NOT_ASSESSED",
        "coverage": {
            "original_points": len(rows),
            "scored_points": scored,
            "ignored_points": len(rows) - scored,
            "missing_points": 0,
        },
        "quality": quality,
        "per_acquisition": {
            name: {
                run: comparison.metric_summary(matrix, classes) for run, matrix in by_run.items()
            }
            for name, by_run in acquisition.items()
        },
        "q1_native_minus_q0": _deltas(quality["q0"], quality["q1_native"]),
        "q1_native_minus_reference": deltas,
        "disagreement_points": disagreements,
        "matched_q1_label_quality_gates": {
            "agreement": agreement,
            "confusion_reference_native_all_points": parity_confusion,
            "label_agreement_pass": label_gate,
            "mean_iou_pass": quality_gate,
            "supported_class_iou_pass": class_gate,
            "verdict": (
                "FAIL"
                if not label_gate or quality_gate is False or not class_gate
                else "INCONCLUSIVE"
                if quality_gate is None
                else "PASS"
            ),
            "discrete_execution_parity": "NOT_ASSESSED",
            "logit_parity": "NOT_ASSESSED",
        },
        "q0_q1_policy_quality_verdict": "NOT_ASSESSED_REQUIRES_CLUSTER_UNCERTAINTY",
        "point_age_q1_native": point_age(rows, windows["q1_native"]),
        "cutoff_reference_inputs": [
            {
                "source_sha256": window["source_sha256"],
                "window_index": window["window_index"],
                "cutoff_ns": window["cutoff_ns"],
                "input_sha256": cutoff_reference(
                    [row for row in rows if row["source_sha256"] == window["source_sha256"]],
                    window["cutoff_ns"],
                )["input_sha256"],
            }
            for window in expected["q1"]
        ],
        "limitations": [
            "Hashed artifacts and validated traces do not attest actual executable inputs "
            "or GPU inference.",
            "Source extraction, declared validation split and model/preprocessing meanings "
            "need independent admission.",
            "Q0 is a whole-source offline reference, never a causal streaming baseline.",
            "No trained-model qualification, logit gate, physical sensor timing or executable "
            "prefix probe is implied.",
            "Decimal trace minima do not replace the frozen model's actual "
            "voxelization parity test.",
        ],
    }


def prefix_causality_probe(execute, prefix, futures):
    """Execute one adapter against removed, appended and changed future inputs.

    execute receives isolated JSON observations and returns committed Q1 prediction
    records containing source_sha256, point_index, window_index, label, cutoff_ns,
    committed_ns. Only committed_ns may vary. EOF partial windows are excluded:
    they are not committed until EOF is declared. Exceptions fail the probe.
    This establishes the sampled execution's prefix property, not universal causality.
    """
    require(
        isinstance(prefix, list)
        and len(prefix) <= MAX_RECORDS
        and prefix
        and len(prefix) % Q1_POINTS == 0,
        "prefix must end at complete Q1 window boundary",
    )
    require(
        isinstance(futures, list) and len(futures) >= 2 and all(future for future in futures),
        "appended and altered nonempty futures required",
    )
    require(
        len(futures) <= 8
        and all(
            isinstance(future, list) and len(prefix) + len(future) <= MAX_RECORDS
            for future in futures
        ),
        "prefix probe work limit exceeded",
    )
    require(
        canonical_hash(futures[0]) != canonical_hash(futures[1]), "future perturbations identical"
    )
    cutoff = max(row["available_ns"] for row in prefix)
    require(
        all(row["available_ns"] > cutoff for future in futures for row in future),
        "future overlaps committed cutoff",
    )
    identities = [_identity(row) for row in prefix]
    identity_set = set(identities)
    require(len(identity_set) == len(prefix), "duplicate prefix source/index")
    require(len({row["source_sha256"] for row in prefix}) == 1, "probe requires one source prefix")
    source = prefix[0]["source_sha256"]
    validate_observations(prefix, [{"sha256": source, "points": len(prefix)}])
    oracle = build_reference_windows(prefix, "q1")
    expected_window = {
        i: window for window in oracle for i in range(window["begin"], window["end"])
    }
    for future in futures:
        validate_observations(
            prefix + future, [{"sha256": source, "points": len(prefix) + len(future)}]
        )
    canonical_outputs = []
    for future in [[], *futures]:
        inputs = comparison.strict_json(json.dumps(prefix + future, allow_nan=False))
        outputs = execute(inputs)
        require(
            isinstance(outputs, list) and len(outputs) <= MAX_RECORDS,
            "bounded probe output list required",
        )
        committed = []
        seen = set()
        for row in outputs:
            require(
                isinstance(row, dict)
                and set(row)
                == {
                    "source_sha256",
                    "point_index",
                    "window_index",
                    "label",
                    "cutoff_ns",
                    "committed_ns",
                },
                "invalid prefix output fields",
            )
            if _identity(row) not in identity_set:
                continue
            require(_identity(row) not in seen, "duplicate or revised prefix output")
            require(
                timestamp(row["cutoff_ns"])
                and row["cutoff_ns"] <= cutoff
                and timestamp(row["committed_ns"])
                and row["committed_ns"] >= row["cutoff_ns"],
                "prefix output sees future or invalid commit clock",
            )
            require(
                all(integer(row[key]) for key in ("point_index", "window_index", "label")),
                "invalid discrete prefix output",
            )
            expected = expected_window[row["point_index"]]
            require(
                row["window_index"] == expected["window_index"]
                and row["cutoff_ns"] == expected["cutoff_ns"],
                "prefix window/cutoff mismatch",
            )
            seen.add(_identity(row))
            committed.append({key: value for key, value in row.items() if key != "committed_ns"})
        require(
            [_identity(row) for row in committed] == identities, "prefix coverage/order mismatch"
        )
        canonical_outputs.append(canonical_hash(committed))
    require(
        len(set(canonical_outputs)) == 1, "prefix causality failed: future changed committed output"
    )
    return {
        "status": "SAMPLED_EXECUTION_PREFIX_PASS",
        "cases": len(canonical_outputs),
        "prefix_input_sha256": canonical_hash(prefix),
        "future_input_sha256": [canonical_hash(future) for future in [[], *futures]],
        "committed_output_sha256": canonical_outputs[0],
        "prefix_points": len(prefix),
        "scope": "Supplied executable adapter and these perturbations only; not GPU qualification.",
    }


def local_lattice(integer_xyz, scales, offsets, anchor, cell_size):
    """Independent exact LAS integer/scale oracle for a separately versioned local lattice.

    This is not the historical float32/rint model preprocessing. Source and anchor
    translation invariance here never retroactively qualifies that frozen baseline.
    """
    require(
        len(integer_xyz) == len(scales) == len(offsets) == len(anchor) == 3, "three axes required"
    )
    require(all(integer(value) for value in integer_xyz), "LAS integer coordinates required")
    scales, offsets, anchor = (
        [_decimal(value) for value in values] for values in (scales, offsets, anchor)
    )
    cell_size = _decimal(cell_size)
    require(
        cell_size > 0 and all(scale > 0 for scale in scales),
        "positive scales and cell size required",
    )
    return [
        (value * Fraction(scale) + Fraction(offset) - Fraction(origin)) // Fraction(cell_size)
        for value, scale, offset, origin in zip(integer_xyz, scales, offsets, anchor, strict=True)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "seal", "evaluate"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    manifest = comparison.load_spec(args.manifest)
    require(args.out.resolve() != args.manifest.resolve(), "output aliases manifest")
    require(
        all(
            args.out.resolve() != artifact_path(args.base, entry).resolve()
            for entry in manifest["artifacts"].values()
        ),
        "output aliases artifact",
    )
    if args.action == "freeze":
        report = {"protocol_sha256": freeze_protocol(manifest, args.base)}
    elif args.action == "seal":
        report = seal_manifest(manifest, args.base)
    else:
        require(
            args.manifest_sha256 is not None, "independently retained --manifest-sha256 required"
        )
        require(
            args.protocol_sha256 is not None, "independently retained --protocol-sha256 required"
        )
        report = evaluate(manifest, args.base, args.manifest_sha256, args.protocol_sha256)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "manifest_sha256": report.get("manifest_sha256"),
                "protocol_sha256": report["protocol_sha256"],
                "action": args.action,
            }
        )
    )
    return 1 if report.get("matched_q1_label_quality_gates", {}).get("verdict") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
