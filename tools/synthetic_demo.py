# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# Developer and project lead: Tuomas Pietilä.
# AI-assisted software engineering, including OpenAI Codex.
"""Run the model-free native LAZ preservation demonstration."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synthetic_laz import create  # noqa: E402
from verify_laz import verify  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--writer", type=Path, default=Path("build/synrex-laz-labels"))
parser.add_argument("--out", type=Path, default=Path("build/synthetic-demo"))
parser.add_argument("--points", type=int, default=50013)
args = parser.parse_args()
source, labels = create(args.out, args.points)
output = args.out / "synthetic.classified.laz"
subprocess.run([str(args.writer.resolve()), str(source), str(labels), str(output)], check=True)
print(json.dumps(verify(source, output, labels), indent=2))
