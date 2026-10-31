# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Paired acquisition-cluster bootstrap for identity-bound comparison summaries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "synrex_comparison_uncertainty", Path(__file__).with_name("runtime_comparison.py")
)
comparison = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparison)

SCHEMA = "synrex.runtime_uncertainty.v1"
MAX_MATRIX_ENTRIES = 2_000_000
MAX_BOOTSTRAP_VALUES = 1_000_000
MAX_WORK = 50_000_000
require = comparison.require
integer = comparison.integer


def population(scenes):
    """The plan binds source population and assignments before measurement."""
    return [{key: scene[key] for key in (
        "scene_id", "cluster_id", "source_sha256"
    )} for scene in scenes]


def matrix(value, size):
    require(isinstance(value, list) and len(value) == size, "invalid confusion rows")
    for row in value:
        require(isinstance(row, list) and len(row) == size, "invalid confusion columns")
        require(all(integer(n) and 0 <= n <= 2**53 for n in row), "invalid confusion count")
    require(sum(map(sum, value)) <= 2**53, "confusion count exceeds exact numeric range")


def validate(spec):
    require(isinstance(spec, dict) and spec.get("schema") == SCHEMA, "unsupported schema")
    classes = spec.get("classes")
    require(isinstance(classes, list) and 1 <= len(classes) <= 256, "invalid classes")
    ids, names = [], []
    for row in classes:
        require(isinstance(row, dict) and set(row) == {"id", "name"}, "invalid class")
        require(integer(row["id"]) and 0 <= row["id"] <= 255, "invalid class id")
        require(comparison.nonempty(row["name"]), "invalid class name")
        ids.append(row["id"])
        names.append(row["name"])
    require(len(set(ids)) == len(ids) and len(set(names)) == len(names), "duplicate class")
    scenes = spec.get("scenes")
    require(isinstance(scenes, list) and scenes, "nonempty scenes required")
    size = len(classes)
    require(2 * len(scenes) * size * size <= MAX_MATRIX_ENTRIES, "matrix memory limit")
    seen, source_seen = set(), set()
    for scene in scenes:
        require(isinstance(scene, dict) and set(scene) == {
            "scene_id", "cluster_id", "source_sha256", "evidence_sha256",
            "reference_confusion", "candidate_confusion"
        }, "invalid scene fields")
        require(comparison.nonempty(scene["scene_id"])
                and comparison.nonempty(scene["cluster_id"]), "scene/cluster id required")
        require(scene["scene_id"] not in seen, "duplicate scene")
        seen.add(scene["scene_id"])
        sources = scene["source_sha256"]
        require(isinstance(sources, list) and sources, "source identities required")
        for source in sources:
            comparison.sha256(source)
            require(source not in source_seen, "source reused across scenes")
            source_seen.add(source)
        comparison.sha256(scene["evidence_sha256"])
        a, b = scene["reference_confusion"], scene["candidate_confusion"]
        matrix(a, size)
        matrix(b, size)
        require([sum(row) for row in a] == [sum(row) for row in b],
                "paired reference truth support mismatch")
    plan = spec.get("plan")
    require(isinstance(plan, dict) and set(plan) == {
        "replicates", "seed", "confidence", "minimum_clusters",
        "population_sha256", "taxonomy_sha256"
    }, "invalid frozen analysis plan")
    require(integer(plan["replicates"]) and 100 <= plan["replicates"] <= 100_000,
            "replicates must be 100..100000")
    require(integer(plan["seed"]) and 0 <= plan["seed"] < 2**64, "invalid seed")
    require(type(plan["confidence"]) in (int, float)
            and 0 < plan["confidence"] < 1, "invalid confidence")
    require(integer(plan["minimum_clusters"]) and plan["minimum_clusters"] >= 2,
            "minimum_clusters must be at least two")
    require(plan["population_sha256"] == comparison.canonical_hash(population(scenes)),
            "frozen population/cluster assignment mismatch")
    require(plan["taxonomy_sha256"] == comparison.canonical_hash(classes),
            "frozen taxonomy mismatch")
    count = len({scene["cluster_id"] for scene in scenes})
    require(plan["replicates"] * (size + 3) <= MAX_BOOTSTRAP_VALUES,
            "bootstrap memory limit")
    require(plan["replicates"] * ((count + 1) * size * size + size) <= MAX_WORK,
            "bootstrap work limit")
    return classes, scenes, plan


def add(target, value):
    for row, other in zip(target, value, strict=True):
        for i, count in enumerate(other):
            row[i] += count


def empty(size):
    return [[0] * size for _ in range(size)]


def deltas(a, b, classes):
    scores = [comparison.metric_summary(m, classes) for m in (a, b)]
    values = {}
    for key in ("accuracy", "mean_iou_all_classes", "macro_f1_all_classes"):
        x, y = [s[key] for s in scores]
        values[key] = y - x if x is not None and y is not None else None
    for i, row in enumerate(classes):
        x, y = [s["classes"][i]["iou"] for s in scores]
        values[f"class_{row['id']}_iou"] = y - x if x is not None and y is not None else None
    return values, scores


def interval(values, replicates, confidence):
    valid = sorted(v for v in values if v is not None)
    # Never silently condition on bootstrap samples in which a metric is defined.
    if len(valid) != replicates:
        return None
    alpha = (1 - confidence) / 2
    return [valid[max(0, math.ceil(p * replicates) - 1)] for p in (alpha, 1 - alpha)]


def analyze(spec):
    """Resample paired acquisitions uniformly; preserve all scenes within each draw.

    Confusions/evidence hashes are supplied by an upstream evaluator. This function
    verifies their shape and pairing, not original-source/truth or model execution.
    """
    spec = comparison.strict_json(json.dumps(spec, allow_nan=False))
    classes, scenes, plan = validate(spec)
    size = len(classes)
    clusters = {}
    for scene in scenes:
        pair = clusters.setdefault(scene["cluster_id"], [empty(size), empty(size)])
        for target, name in zip(pair, ("reference_confusion", "candidate_confusion"), strict=True):
            add(target, scene[name])
    pairs = list(clusters.values())
    pooled = [empty(size), empty(size)]
    for pair in pairs:
        for target, value in zip(pooled, pair, strict=True):
            add(target, value)
    require(sum(map(sum, pooled[0])) <= 2**53, "pooled count exceeds exact numeric range")
    observed, scores = deltas(*pooled, classes)
    samples = {key: [] for key in observed}
    rng = random.Random(plan["seed"])
    n = len(pairs)
    require(n * max(sum(map(sum, pair[0])) for pair in pairs) <= 2**53,
            "possible bootstrap count exceeds exact numeric range")
    if n >= 2:
        for _ in range(plan["replicates"]):
            totals = [empty(size), empty(size)]
            for _ in range(n):
                pair = pairs[rng.randrange(n)]
                for target, value in zip(totals, pair, strict=True):
                    add(target, value)
            result, _ = deltas(*totals, classes)
            for key, value in result.items():
                samples[key].append(value)
    metrics = {}
    enough = n >= plan["minimum_clusters"]
    for key, value in observed.items():
        draws = samples[key]
        bounds = interval(draws, plan["replicates"], plan["confidence"]) if n >= 2 else None
        metrics[key] = {
            "candidate_minus_reference": value,
            "percentile_interval": bounds,
            "undefined_replicates": sum(v is None for v in draws),
            "status": "REPORTED" if enough and bounds is not None
            else "INCONCLUSIVE",
        }
    return {
        "schema": SCHEMA, "qualification_status": "NOT_ASSESSED",
        "spec_sha256": comparison.canonical_hash(spec),
        "plan_sha256": comparison.canonical_hash(plan),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison_scorer_sha256": hashlib.sha256(
            Path(comparison.__file__).read_bytes()).hexdigest(),
        "plan": plan, "scenes": len(scenes), "clusters": n,
        "population": population(scenes), "metrics": metrics,
        "upstream_evidence": [{"scene_id": s["scene_id"], "sha256": s["evidence_sha256"]}
                              for s in scenes],
        "pooled_quality": dict(zip(("reference", "candidate"), scores, strict=True)),
        "class_support": [{
            **row, "points": scores[0]["classes"][i]["support"],
            "scenes": sum(sum(s["reference_confusion"][i]) > 0 for s in scenes),
            "clusters": sum(sum(p[0][i]) > 0 for p in pairs), "objects": None,
        } for i, row in enumerate(classes)],
        "method": {
            "unit": "uniform acquisition-cluster resampling; same paired draws for both runs",
            "metric": "pooled confusion within every draw; candidate minus reference in fractions",
            "interval": "two-sided empirical percentile, inverse empirical CDF (nearest rank)",
            "absent_class": ("all-class macro uses zero; undefined per-class IoU draws "
                             "invalidate its interval"),
            "evaluated_replicates": plan["replicates"] if n >= 2 else 0,
        },
        "limitations": [
            "Input confusions, scene/acquisition membership and evidence hashes are declarations.",
            "No proof of truth lineage, preregistration time, cluster independence or execution.",
            "Intervals are marginal, not simultaneous; no quality noninferiority decision is made.",
            "minimum_clusters is a frozen reporting threshold, not proof of statistical adequacy.",
            "Rare object and physical temporal stability support are unavailable.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.spec.resolve() != args.out.resolve(), "output aliases input")
    result = analyze(comparison.load_spec(args.spec))
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
