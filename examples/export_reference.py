# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Original deterministic 1x1x1 Conv3D fixture. Export needs ONNX, never PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def export(spec_path: Path, output: Path) -> None:
    spec = json.loads(spec_path.read_text())
    if len(spec["features"]) != 2 or len(spec["las_classes"]) != 3:
        raise ValueError("reference constants require two features and three classes")
    dtype = np.float16 if spec["input"]["dtype"] == "float16" else np.float32
    # Scores [1 - 2*i, 0.6, -1 + 2*i] at occupied voxels.
    # Lowest channel wins ties. Low/mid/high intensity exercise all three classes.
    weights = np.array([[1, -2], [0.6, 0], [-1, 2]], dtype=dtype).reshape(3, 2, 1, 1, 1)
    bias = np.zeros(3, dtype=dtype)
    name_in, name_out = spec["input"]["name"], spec["output"]["name"]
    conv_out = name_out if spec["input"]["dtype"] == spec["output"]["dtype"] else "conv_result"
    nodes = [helper.make_node("Conv", [name_in, "W", "B"], [conv_out], kernel_shape=[1, 1, 1])]
    types = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16}
    if conv_out != name_out:
        nodes.append(
            helper.make_node("Cast", [conv_out], [name_out], to=types[spec["output"]["dtype"]])
        )
    graph = helper.make_graph(
        nodes,
        "synrex-original-synthetic-conv3d-v1",
        [
            helper.make_tensor_value_info(
                name_in, types[spec["input"]["dtype"]], spec["input"]["shape"]
            )
        ],
        [
            helper.make_tensor_value_info(
                name_out, types[spec["output"]["dtype"]], spec["output"]["shape"]
            )
        ],
        [numpy_helper.from_array(weights, "W"), numpy_helper.from_array(bias, "B")],
    )
    model = helper.make_model(
        graph,
        producer_name="SynRex original synthetic fixture",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    model.doc_string = "MIT; original analytic constants; no training data or restricted weights"
    onnx.checker.check_model(model, full_check=True)
    with output.open("xb") as stream:
        stream.write(model.SerializeToString(deterministic=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path(__file__).with_name("dense_voxel.json"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    export(args.spec, args.out)
