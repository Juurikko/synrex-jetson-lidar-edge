# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Derive application throughput from immutable receipts without GPU execution."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipts", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads(a.receipts.read_text())
    old = json.loads(a.historical.read_text())
    rows = []
    for r in receipt["records"]:
        cold = r["mean_cold_seconds"]
        rows.append(
            dict(
                source=r["source"],
                points=r["points"],
                cold_launch_to_exit_seconds=r["mean_process_launch_to_exit_seconds"],
                persistent_seconds=r["mean_persistent_seconds"]["wall_seconds"],
                laz_decode_seconds=cold["laz_decode_cpu"],
                geometry_seconds=cold["geometry_cpu"],
                cuda_preparation_seconds=cold["cuda_tensor_preparation"],
                inference_seconds=cold["tensorrt_inference"],
                cuda_argmax_mapping_reconstruction_seconds=cold["cuda_postprocessing"],
                laz_encoding_seconds=cold["laz_encoding_and_preservation"],
                initialization_initial_upload_seconds=cold["gpu_initialization_and_initial_upload"],
                tracked_peak_requested_cuda_bytes=r["persistent"][
                    "tracked_peak_requested_cuda_bytes"
                ],
                sampled_device_max_mib=max(
                    c["sampled_max_device_used_mib"] for c in r["cold"]["records"]
                ),
                complete_points_per_second=r["points"] / r["mean_process_launch_to_exit_seconds"],
            )
        )
    points = sum(r["points"] for r in rows)
    cold = sum(r["cold_launch_to_exit_seconds"] for r in rows)
    warm = sum(r["persistent_seconds"] for r in rows)
    result = dict(
        status="PASS",
        method=(
            "sum(points)/sum(mean per-scene duration); equal repeat counts; never "
            "arithmetic mean of per-scene throughput"
        ),
        input_receipts_sha256=hashlib.sha256(a.receipts.read_bytes()).hexdigest(),
        historical_receipt_sha256=hashlib.sha256(a.historical.read_bytes()).hexdigest(),
        scenes=rows,
        total_points=points,
        aggregate_process_seconds=cold,
        aggregate_complete_points_per_second=points / cold,
        aggregate_cached_processing_points_per_second_excludes_io=points / warm,
        historical_agx_segformer_largest_points_per_second=old["scenes"][-1]["points_per_second"],
        historical_agx_segformer_aggregate_points_per_second=old["aggregate_points_per_second"],
        comparison=(
            "Descriptive observations only: different GPU, model, data, ontology, voxel "
            "representation, runtime and timing boundaries; not controlled speedup "
            "evidence"
        ),
        bottlenecks=(
            "Cold engine/context/loader/teardown overhead; larger scenes LAZ "
            "decode/encoding. TensorRT inference and CUDA reconstruction are small "
            "fractions of complete latency."
        ),
        memory_scope=(
            "Tracked requested CUDA allocation peak excludes context/untracked vendor "
            "allocations; device-wide sampled maximum includes desktop baseline and may "
            "miss transient peak"
        ),
    )
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
