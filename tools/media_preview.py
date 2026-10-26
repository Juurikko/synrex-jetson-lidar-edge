# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# Developer and project lead: Tuomas Pietilä.
# AI-assisted software engineering, including OpenAI Codex.
"""Create a small GIF excerpt from the accepted R1 MP4; keep the MP4 untouched."""

import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

root = Path(__file__).resolve().parents[1]
video = root / "media/synrex-jetson-lidar-r1.mp4"
gif = root / "media/preview.gif"
filters = (
    "[0:v]fps=6,scale=360:450:flags=lanczos,split[a][b];"
    "[a]palettegen=max_colors=96:stats_mode=diff[p];"
    "[b][p]paletteuse=dither=bayer:bayer_scale=3"
)
subprocess.run(
    [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-filter_complex",
        filters,
        "-t",
        "9",
        "-loop",
        "0",
        str(gif),
    ],
    check=True,
)
manifest_path = root / "evidence/media_manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest[gif.name] = {
    "sha256": hashlib.sha256(gif.read_bytes()).hexdigest(),
    "bytes": gif.stat().st_size,
    "origin": "First 9 seconds of accepted R1 MP4; 360x450 GIF preview at 6 fps, 96-color palette",
    "source_mp4_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
    "illustrative_only": True,
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
