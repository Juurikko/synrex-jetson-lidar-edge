# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Rescore sealed original-order semantic sidecars against raw LAS/LAZ labels on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from . import runtime_comparison as comparison
except ImportError:
    import runtime_comparison as comparison

SCHEMA = "synrex.runtime_sealed_outputs.v1"
MAX_SIDECAR_BYTES = 64 * 1024 * 1024
MAX_SOURCE_POINTS = 2_000_000
CHUNK_POINTS = 65_536
require = comparison.require


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def sealed_json(artifact, limit):
    require(isinstance(artifact, dict) and set(artifact) == {"path", "sha256"},
            "artifact requires path and SHA256")
    require(comparison.nonempty(artifact["path"]), "artifact path required")
    comparison.sha256(artifact["sha256"])
    with Path(artifact["path"]).open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "sealed JSON artifact exceeds byte bound")
    require(hashlib.sha256(raw).hexdigest() == artifact["sha256"], "artifact SHA256 mismatch")
    return comparison.strict_json(raw)


def validate_spec(spec):
    require(isinstance(spec, dict) and set(spec) == {
        "schema", "comparison", "source_las_to_label", "runs", "sources",
    } and spec["schema"] == SCHEMA, "invalid sealed-output specification")
    classes, ignored, ground = comparison.validate_spec(spec["comparison"])
    require(spec["comparison"]["track"] == "model_policy_quality",
            "retained sidecar rescoring is quality/compatibility analysis, not runtime parity")
    index = {row["id"]: i for i, row in enumerate(classes)}
    mapping = spec["source_las_to_label"]
    require(isinstance(mapping, list) and mapping, "explicit raw source label mapping required")
    source_mapping = {}
    for row in mapping:
        require(isinstance(row, dict) and set(row) == {"las_code", "label"},
                "invalid raw source label mapping")
        code, label = row["las_code"], row["label"]
        require(type(code) is int and 0 <= code <= 255 and code not in source_mapping,
                "invalid or duplicate raw LAS code")
        require(type(label) is int and label in set(index) | set(ignored),
                "raw source label maps outside scoring taxonomy")
        source_mapping[code] = label
    runs = spec["runs"]
    require(isinstance(runs, dict) and set(runs) == {"reference", "candidate"},
            "two retained runs required")
    for name, run in runs.items():
        require(isinstance(run, dict) and set(run) == {"package_manifest", "channel_to_label"},
                "invalid retained run")
        labels = run["channel_to_label"]
        require(isinstance(labels, list) and labels and len(labels) <= 256
                and all(type(v) is int and v in index for v in labels)
                and len(set(labels)) == len(labels), "explicit bijective channel mapping required")
        package = sealed_json(run["package_manifest"], comparison.MAX_JSON_BYTES)
        require(isinstance(package, dict) and isinstance(package.get("contract"), dict),
                "package contract required")
        projection = package["contract"].get("las_classes")
        require(isinstance(projection, list) and len(projection) == len(labels)
                and all(type(v) is int and 0 <= v <= 255 for v in projection),
                "package LAS projection mismatch")
        artifact = package.get("artifacts", {}).get("engine", {})
        comparison.sha256(artifact.get("sha256"))
        declared_engine = spec["comparison"][name]["provenance"]["engine_sha256"]
        require(declared_engine is not None and artifact["sha256"] == declared_engine,
                "declared engine differs from sealed package manifest")
    rows = spec["sources"]
    require(isinstance(rows, list) and len(rows) == len(spec["comparison"]["sources"]),
            "source roster mismatch")
    for row, source in zip(rows, spec["comparison"]["sources"], strict=True):
        require(isinstance(row, dict) and set(row) == {"path", "reference", "candidate"}
                and comparison.nonempty(row["path"]), "invalid source artifacts")
        require(0 < source["points"] <= MAX_SOURCE_POINTS,
                "retained source point count exceeds bounded per-source adapter")
    return classes, ignored, ground, index, source_mapping


def sidecar(artifact, source, run, projection):
    info = sealed_json(artifact, MAX_SIDECAR_BYTES)
    require(isinstance(info, dict) and set(info) == {
        "schema", "index_order", "semantic_id_definition", "source_sha256",
        "model_manifest_sha256", "point_count", "las_classes", "semantic_ids",
        "mapped_labels_sha256",
    }, "invalid semantic sidecar fields")
    require(info["schema"] == "synrex.point_semantics.v1"
            and info["index_order"] == "original source point record order"
            and info["semantic_id_definition"]
            == "zero-based model output channel; no LAS class collapse",
            "unsupported semantic sidecar order or definition")
    require(info["source_sha256"] == source["sha256"]
            and type(info["point_count"]) is int and info["point_count"] == source["points"],
            "sidecar source/count mismatch")
    require(info["model_manifest_sha256"] == run["package_manifest"]["sha256"],
            "sidecar package binding mismatch")
    require(isinstance(info["las_classes"], list)
            and all(type(v) is int for v in info["las_classes"])
            and info["las_classes"] == projection, "sidecar projection mismatch")
    values = info["semantic_ids"]
    require(isinstance(values, list) and len(values) == source["points"]
            and all(type(v) is int and 0 <= v < len(projection) for v in values),
            "invalid original-order semantic vector")
    comparison.sha256(info["mapped_labels_sha256"])
    mapped = bytes(projection[v] for v in values)
    require(hashlib.sha256(mapped).hexdigest() == info["mapped_labels_sha256"],
            "sidecar mapped-label digest mismatch")
    return values


def evaluate(spec):
    """Decode raw labels in chunks; keep at most one source's bounded sidecars in memory.

    This does not execute engines or verify inference/preservation/clock provenance.
    All package, scoring and historical execution metadata is preserved as declared.
    """
    import laspy
    import numpy as np

    spec = comparison.strict_json(json.dumps(spec, allow_nan=False))
    classes, ignored, ground, indices, mapping = validate_spec(spec)
    count = len(classes)
    matrices = {name: np.zeros((count, count), dtype=np.int64)
                for name in ("reference", "candidate")}
    transition = np.zeros((count, count), dtype=np.int64)
    source_rows = []
    ignored_total = total = different = 0
    projections = {
        name: sealed_json(run["package_manifest"], comparison.MAX_JSON_BYTES)["contract"][
            "las_classes"]
        for name, run in spec["runs"].items()
    }
    for artifacts, source in zip(spec["sources"], spec["comparison"]["sources"], strict=True):
        require(digest(artifacts["path"]) == source["sha256"], "raw source SHA256 mismatch")
        values = {
            name: sidecar(artifacts[name], source, run, projections[name])
            for name, run in spec["runs"].items()
        }
        vectors = {
            name: np.asarray([indices[run["channel_to_label"][v]] for v in values[name]],
                             dtype=np.int64)
            for name, run in spec["runs"].items()
        }
        raw_semantic_hashes = {
            name: hashlib.sha256(bytes(vector)).hexdigest() for name, vector in values.items()
        }
        del values
        local = {name: np.zeros((count, count), dtype=np.int64) for name in matrices}
        truth_hash = hashlib.sha256()
        decoded = 0
        local_ignored = local_different = 0
        with laspy.open(artifacts["path"], laz_backend=laspy.LazBackend.Lazrs) as reader:
            require(reader.header.point_count == source["points"],
                    "raw header point count mismatch")
            for chunk in reader.chunk_iterator(CHUNK_POINTS):
                raw = np.asarray(chunk.classification, dtype=np.uint8)
                require(decoded + len(raw) <= source["points"], "extra decoded raw source points")
                truth_hash.update(raw.tobytes())
                require(all(int(v) in mapping for v in np.unique(raw)), "unmapped raw source label")
                truth = np.asarray([mapping[int(v)] for v in raw], dtype=np.int64)
                valid = ~np.isin(truth, ignored)
                truth_indices = np.asarray([indices[int(v)] for v in truth[valid]], dtype=np.int64)
                a, b = (vectors[name][decoded:decoded + len(raw)]
                        for name in ("reference", "candidate"))
                transition += np.bincount(a * count + b,
                                          minlength=count * count).reshape(count, count)
                local_different += int(np.count_nonzero(a != b))
                local_ignored += int(np.count_nonzero(~valid))
                for name, predictions in (("reference", a), ("candidate", b)):
                    local[name] += np.bincount(truth_indices * count + predictions[valid],
                                               minlength=count * count).reshape(count, count)
                decoded += len(raw)
        require(decoded == source["points"], "decoded raw source count mismatch")
        require(digest(artifacts["path"]) == source["sha256"], "raw source changed during scoring")
        # Rehash small sealed metadata/output artifacts after the decoder finishes too.
        for name, run in spec["runs"].items():
            for artifact in (artifacts[name], run["package_manifest"]):
                require(digest(artifact["path"]) == artifact["sha256"],
                        "sealed artifact changed during scoring")
            matrices[name] += local[name]
        source_rows.append({
            "source_sha256": source["sha256"], "points": decoded,
            "raw_classification_sha256": truth_hash.hexdigest(),
            "semantic_channel_vector_sha256": raw_semantic_hashes,
            "sidecar_sha256": {name: artifacts[name]["sha256"] for name in matrices},
            "ignored_points": local_ignored, "label_disagreements": local_different,
            "quality": {name: comparison.metric_summary(matrix.tolist(), classes)
                        for name, matrix in local.items()},
        })
        total += decoded
        ignored_total += local_ignored
        different += local_different
    scores = {name: comparison.metric_summary(matrix.tolist(), classes)
              for name, matrix in matrices.items()}
    return {
        "schema": SCHEMA, "measurement_status": "COMPLETE",
        "qualification_status": "NOT_ASSESSED",
        "execution": "CPU_REEVALUATION_RETAINED_SIDECARS_NO_INFERENCE",
        "spec": spec, "spec_sha256": comparison.canonical_hash(spec),
        "scorer_sha256": digest(__file__), "comparison_scorer_sha256": digest(comparison.__file__),
        "decoder": {"name": "laspy.LazBackend.Lazrs", "laspy_version": laspy.__version__,
                    "numpy_version": np.__version__, "chunk_points": CHUNK_POINTS,
                    "per_source_point_limit": MAX_SOURCE_POINTS,
                    "sidecar_byte_limit": MAX_SIDECAR_BYTES},
        "coverage": {"original_points": total, "scored_points": total - ignored_total,
                     "ignored_points": ignored_total, "missing_points": 0},
        "quality": scores,
        "candidate_minus_reference": {
            metric: scores["candidate"][metric] - scores["reference"][metric]
            if scores["candidate"][metric] is not None else None
            for metric in ("accuracy", "mean_iou_all_classes", "macro_f1_all_classes")
        },
        "label_disagreement": {"points": different, "agreement": (total - different) / total,
                               "confusion_reference_candidate": transition.tolist()},
        "ground_proxy": {name: comparison.ground_proxy(matrix.tolist(), classes, ground)
                         for name, matrix in matrices.items()},
        "sources": source_rows,
        "limitations": [
            "Retained channel vectors; no engine, logits or current runtime execution.",
            "Manifest rehashed; engine/weights and lineage need upstream verification.",
            "Raw labels decoded for scoring only; no features, training or selection.",
            "No output-LAZ, physical EOF, streaming, capacity or terrain qualification.",
            "Frozen scoring/source mappings and execution metadata remain supplied declarations.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "output already exists")
    result = evaluate(comparison.load_spec(args.spec))
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"measurement_status": result["measurement_status"],
                      "execution": result["execution"], "coverage": result["coverage"]}))


if __name__ == "__main__":
    main()
