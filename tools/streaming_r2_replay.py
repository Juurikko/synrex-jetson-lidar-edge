# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Prepare a licensed replay and independently verify recorded native map events.

Offline tooling only. No neural inference, camera motion, map labels or confidence
values are synthesized here. Replay timestamps are explicitly emulated, not LAS GPS.
"""

import argparse
import hashlib
import json
import math
from collections import OrderedDict, deque
from pathlib import Path

import laspy
import numpy as np


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def points(source):
    cloud = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    return cloud, xyz


def prepare(source, out):
    cloud, xyz = points(source)
    if cloud.header.parse_crs().to_epsg() != 2154:
        raise ValueError("Showcase requires genuine embedded EPSG:2154; no implicit assignment")
    out.mkdir(parents=True, exist_ok=False)
    epoch = 1700000000000000000
    sequence = 0
    with (out / "input.jsonl").open("x") as f:
        for frame, start in enumerate(range(0, len(cloud), 16384)):
            end = min(start + 16384, len(cloud))
            frame_start = epoch + frame * 100000000
            for chunk, first in enumerate(range(start, end, 4096)):
                records = []
                for i in range(first, min(first + 4096, end)):
                    rn, rc = int(cloud.return_number[i]), int(cloud.number_of_returns[i])
                    records.append(
                        dict(
                            id=i,
                            timestamp_ns=frame_start + i - start,
                            xyz=xyz[i].tolist(),
                            intensity=int(cloud.intensity[i]),
                            return_number=rn if rn and rc else None,
                            return_count=rc if rn and rc else None,
                        )
                    )
                row = dict(
                    schema="synrex.point_stream.v1",
                    sensor_id="IGN-recorded-scene",
                    stream_id="showcase-C-replay",
                    sequence=sequence,
                    frame_id=frame,
                    frame_start_ns=frame_start,
                    frame_end_ns=frame_start + end - start - 1,
                    chunk_index=chunk,
                    chunk_count=math.ceil((end - start) / 4096),
                    frame_points=end - start,
                    units="metres",
                    coordinate_frame="projected",
                    crs="EPSG:2154",
                    clock="unix_ns",
                    taxonomy="synrex.fractal7.v1",
                    points=records,
                )
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                sequence += 1
    save(
        out / "input-provenance.json",
        dict(
            schema="synrex.streaming_r2.input.v1",
            status="PASS",
            source_name=source.name,
            source_sha256=sha(source),
            input_sha256=sha(out / "input.jsonl"),
            points=len(cloud),
            crs="EPSG:2154 embedded source WKT",
            window_points=16384,
            chunk_points=4096,
            timestamps="Emulated UNIX-ns replay clock; not original sensor GPS time",
            ordering="Unmodified LAS record order; no coordinate translation or fabricated flight",
            license="IGN/FRACTAL Etalab-2.0; revision 36e00245ef92fd180f655f6ee410f2e935693cc2",
        ),
    )


def verify(source, recording, model):
    cloud, xyz = points(source)
    provenance = json.loads((recording / "input-provenance.json").read_text())
    receipt = json.loads((recording / "native-receipt.json").read_text())
    assert provenance["source_sha256"] == sha(source)
    assert provenance["input_sha256"] == receipt["input_sha256"] == sha(recording / "input.jsonl")
    assert receipt["status"] == "PASS" and receipt["events_sha256"] == sha(
        recording / "events.jsonl"
    )
    assert receipt["emitted_points"] == receipt["freshly_classified_points"] == len(cloud)
    manifest_sha = sha(model / "manifest.json")
    assert manifest_sha == "2d1206be62385669477a180d98c1b330c1caf40d516625f64682284012f7a325"
    manifest = json.loads((model / "manifest.json").read_text())
    assert sha(model / "model.engine") == manifest["artifacts"]["engine"]["sha256"]
    session = receipt["session"]
    assert session["backend"] == "CUDA/TensorRT"
    assert session["manifest_sha256"] == manifest_sha
    assert session["artifacts"]["engine"]["sha256"] == sha(model / "model.engine")
    assert session["execution_contexts"] == 1 and session["successful_enqueue_calls"] > 0
    assert (
        receipt["geometry_only_ingested_points"] == receipt["inferred_classes_reused_points"] == 0
    )
    codes = [1, 2, 200, 6, 9, 17, 201]
    oracle, observed = OrderedDict(), {}
    expected = deque()
    labels = np.full(len(cloud), 255, np.uint8)
    seen = set()
    snapshots = []
    frames = 0
    delta_count = 0

    def snapshot():
        assert not expected, "missing native deltas"
        keys = sorted(observed)
        assert {k: v for k, v in oracle.items()} == observed
        snapshots.append(
            dict(
                keys=np.asarray(keys, dtype=np.int64).reshape(-1, 3),
                ids=np.asarray([observed[k][0] for k in keys], dtype=np.uint8),
                times=np.asarray([observed[k][1] for k in keys], dtype=np.uint64),
            )
        )

    with (recording / "events.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["schema"] == "synrex.classified_frame.v1":
                if frames:
                    snapshot()
                assert row["mode"] == "all_points_freshly_classified"
                assert row["frame_id"] == frames and row["crs"] == "EPSG:2154"
                assert row["coordinate_frame"] == "projected"
                for p in row["points"]:
                    i = p["id"]
                    assert i == len(seen) and i not in seen and i < len(cloud)
                    seen.add(i)
                    assert p["xyz"] == xyz[i].tolist()
                    assert p["intensity"] == int(cloud.intensity[i])
                    rn, rc = int(cloud.return_number[i]), int(cloud.number_of_returns[i])
                    assert p["return_number"] == (rn if rn and rc else None)
                    assert p["return_count"] == (rc if rn and rc else None)
                    timestamp = 1700000000000000000 + frames * 100000000 + i % 16384
                    assert p["timestamp_ns"] == timestamp
                    semantic = p["semantic_id"]
                    assert type(semantic) is int and 0 <= semantic < 7
                    assert p["model_las_code"] == codes[semantic] and p["confidence"] is None
                    labels[i] = semantic
                    key = tuple(math.floor(v) for v in p["xyz"])
                    previous = oracle.get(key)
                    if previous and timestamp < previous[1]:
                        continue
                    evicted = None
                    if previous is None and len(oracle) == 65536:
                        evicted, _ = oracle.popitem(last=False)
                    oracle[key] = semantic, timestamp
                    oracle.move_to_end(key)
                    if previous != (semantic, timestamp):
                        expected.append((key, semantic, timestamp, evicted))
                frames += 1
            else:
                assert row["schema"] == "synrex.semantic_delta.v1"
                assert row["mode"] == "map_delta_output" and row["crs"] == "EPSG:2154"
                assert row["confidence"] is None and row["confidence_age_ns"] is None
                key = tuple(row["key"])
                victim = tuple(row["evicted"]) if row["evicted"] is not None else None
                assert expected.popleft() == (key, row["semantic_id"], row["timestamp_ns"], victim)
                if victim is not None:
                    del observed[victim]
                observed[key] = row["semantic_id"], row["timestamp_ns"]
                assert len(observed) <= 65536
                delta_count += 1
    snapshot()
    assert seen == set(range(len(cloud))) and np.all(labels < 7)
    assert len(observed) == receipt["map_resident_voxels"]
    assert delta_count == receipt["map_updates"]
    # Snapshots use only verified received deltas, never reference labels or guessed map states.
    arrays = {}
    for i, snap in enumerate(snapshots):
        arrays.update({f"{name}_{i:03d}": value for name, value in snap.items()})
    if (recording / "snapshots.npz").exists():
        raise FileExistsError("recording verification products must be new")
    np.savez_compressed(recording / "snapshots.npz", **arrays)
    (recording / "predictions.u8").write_bytes(labels.tobytes())
    gt = np.full(len(cloud), 255, np.uint8)
    for semantic, source_codes in enumerate([[1], [2], [3, 4, 5], [6], [9], [17], [64]]):
        gt[np.isin(cloud.classification, source_codes)] = semantic
    valid = gt < 7
    mistakes = np.flatnonzero(valid & (gt != labels))
    save(
        recording / "verification.json",
        dict(
            schema="synrex.streaming_r2.visual_provenance.v1",
            status="PASS",
            source_sha256=sha(source),
            model_manifest_sha256=manifest_sha,
            engine_sha256=sha(model / "model.engine"),
            input_sha256=sha(recording / "input.jsonl"),
            events_sha256=sha(recording / "events.jsonl"),
            receipt_sha256=sha(recording / "native-receipt.json"),
            predictions_sha256=sha(recording / "predictions.u8"),
            snapshots_sha256=sha(recording / "snapshots.npz"),
            frames=frames,
            freshly_classified_source_points=len(cloud),
            coverage=1.0,
            duplicate_points=0,
            dropped_points=0,
            map_deltas=delta_count,
            peak_map_cells=max(len(s["ids"]) for s in snapshots),
            final_map_cells=len(observed),
            semantic_histogram=np.bincount(labels, minlength=7).tolist(),
            scored_reference_points=int(valid.sum()),
            actual_disagreements=int(len(mistakes)),
            disagreement_example_source_ids=mistakes[:32].tolist(),
            map_policy="Capacity 65536; 1 m grid; last accepted observation revision; no TTL",
            confidence="Unavailable; never invented",
            render_timing="Illustrative playback of recorded frames",
            neural_execution="Native recorded fresh frame classifications; no map label reuse",
            successful_neural_enqueues=session["successful_enqueue_calls"],
            execution_contexts=session["execution_contexts"],
            recording_p99_ms=receipt["acceptance_to_output_seconds"]["p99"] * 1000,
            recording_latency_gate=(
                "PASS" if receipt["acceptance_to_output_seconds"]["p99"] <= 0.250 else "FAIL"
            ),
            recording_latency_scope="Includes large JSON event file writes; media correctness only",
            qualification_scope="Short media recording, not a new sustained performance trial",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "verify"])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--recording", type=Path, required=True)
    p.add_argument("--model", type=Path)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.source, a.recording)
    else:
        if a.model is None:
            p.error("verify requires --model")
        verify(a.source, a.recording, a.model)


if __name__ == "__main__":
    main()
