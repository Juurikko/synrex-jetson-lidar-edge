# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Render genuine verified replay outputs with a deterministic CPU 3D point rasterizer.

No generated scenery, guessed classes, fake sensor poses, music or stock media.
Camera movement is illustrative. Data coordinates and semantic IDs stay untouched.
Requires the independently verified streaming_r2_replay.py products, Pillow and FFmpeg.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SECONDS = 1080, 1350, 24, 54
BG, PANEL, WHITE, MUTED, TEAL = "#0a1420", "#101f2d", "#f1f6f8", "#a3b6c7", "#48e0c1"
COLORS = ["#8a9297", "#a77c43", "#2c9b59", "#f2594b", "#357bd9", "#b96ed0", "#f7b62e"]
NAMES = ["Other", "Ground", "Vegetation", "Building", "Water", "Bridge", "Permanent structure"]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rgb(color):
    return tuple(bytes.fromhex(color.lstrip("#")))


class Renderer:
    def __init__(self, source, recording, qgis_image, font, second_scene=None):
        self.source, self.recording, self.qgis_image = source, recording, qgis_image
        self.proof = json.loads((recording / "verification.json").read_text())
        assert self.proof["status"] == "PASS" and self.proof["source_sha256"] == sha(source)
        for name, key in [
            ("predictions.u8", "predictions_sha256"),
            ("snapshots.npz", "snapshots_sha256"),
            ("events.jsonl", "events_sha256"),
        ]:
            assert sha(recording / name) == self.proof[key]
        self.cloud = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
        self.xyz = np.column_stack((self.cloud.x, self.cloud.y, self.cloud.z))
        self.center = (self.xyz.min(axis=0) + self.xyz.max(axis=0)) / 2
        self.labels = np.fromfile(recording / "predictions.u8", np.uint8)
        assert len(self.labels) == len(self.xyz) and np.all(self.labels < 7)
        self.secondary = None
        self.secondary_provenance = None
        if second_scene is not None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from verify_semantics import verify_semantics

            stem = "TEST-0741_6244-006630086"
            original = second_scene / "data/source" / (stem + ".laz")
            classified = second_scene / "data/classified" / (stem + "-classified.laz")
            sidecar = second_scene / "data/semantics" / (stem + "-semantics.json")
            binding = second_scene / "data/semantics" / (stem + "-binding.json")
            model = second_scene / "model/unet.srxmodel"
            assert sha(model / "manifest.json") == self.proof["model_manifest_sha256"]
            verify_semantics(
                original,
                classified,
                sidecar,
                model,
                binding,
                Path(__file__).resolve().parents[1] / "examples/unet/semantic_classes.json",
            )
            cloud = laspy.read(original, laz_backend=laspy.LazBackend.Lazrs)
            assert cloud.header.parse_crs().to_epsg() == 2154
            xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
            labels = np.asarray(json.loads(sidecar.read_text())["semantic_ids"], dtype=np.uint8)
            self.secondary = xyz, labels
            self.secondary_provenance = dict(
                source_name=original.name,
                points=len(xyz),
                source_sha256=sha(original),
                classified_sha256=sha(classified),
                sidecar_sha256=sha(sidecar),
                binding_sha256=sha(binding),
                model_manifest_sha256=sha(model / "manifest.json"),
                origin="Published and independently reverified R4 native predictions",
            )
        self.palette = np.array([rgb(c) for c in COLORS], np.uint8)
        # Explicit deterministic viewing decimation only; every source point was classified.
        self.indices = np.arange(0, len(self.xyz), 3)
        self.gt = np.full(len(self.xyz), 255, np.uint8)
        for i, codes in enumerate([[1], [2], [3, 4, 5], [6], [9], [17], [64]]):
            self.gt[np.isin(self.cloud.classification, codes)] = i
        self.snapshots = np.load(recording / "snapshots.npz", allow_pickle=False)
        self.font = {
            s: ImageFont.truetype(str(font), s)
            for s in [22, 25, 27, 30, 34, 42, 50, 60, 68, 78, 100]
        }
        self.cache = {}
        self.checked_text = set()

    def text(self, image, xy, value, size=30, color=WHITE):
        draw = ImageDraw.Draw(image)
        box = draw.textbbox(xy, value, font=self.font[size])
        assert box[0] >= 0 and box[1] >= 0 and box[2] <= W - 20 and box[3] <= H - 20, (value, box)
        self.checked_text.add((value, size))
        draw.text(xy, value, font=self.font[size], fill=color)

    def base(self, chapter, title, subtitle):
        image = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(image)
        d.line((54, 100, 1026, 100), fill="#294150", width=2)
        self.text(image, (54, 40), "SynRex", 34)
        self.text(image, (680, 48), "STREAMING R2 / REVIEW", 22, TEAL)
        self.text(image, (54, 126), chapter, 25, TEAL)
        for n, line in enumerate(title.split("\n")):
            self.text(image, (50, 169 + n * 79), line, 68)
        self.text(image, (54, 345), subtitle, 27, MUTED)
        return image

    def project(self, xyz, azimuth=-0.85, pitch=0.66, width=972, height=660, center=None):
        p = xyz - (self.center if center is None else center)
        right = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
        forward = np.array(
            [
                -math.sin(azimuth) * math.cos(pitch),
                math.cos(azimuth) * math.cos(pitch),
                -math.sin(pitch),
            ]
        )
        up = np.cross(right, forward)
        depth = 112 + p @ forward
        focal = width * 1.28
        x = width / 2 + (p @ right) * focal / depth
        y = height / 2 - (p @ up) * focal / depth
        return np.column_stack((x, y)), depth

    def cloud_view(
        self, mode="prediction", angle=-0.85, snapshot=None, width=972, height=660, secondary=False
    ):
        view = Image.new("RGB", (width, height), PANEL)
        d = ImageDraw.Draw(view)
        xyz, labels = self.secondary if secondary else (self.xyz, self.labels)
        if secondary:
            assert mode == "prediction" and snapshot is None
        indices = np.arange(0, len(xyz), 3)
        floor_z = xyz[:, 2].min() - 1
        low, high = xyz.min(axis=0), xyz.max(axis=0)
        center = (low + high) / 2
        for axis in [0, 1]:
            for offset in np.arange(math.floor(low[axis] / 10) * 10, high[axis] + 10, 10):
                endpoints = np.array(
                    [[low[0] - 2, low[1] - 2, floor_z], [high[0] + 2, high[1] + 2, floor_z]]
                )
                endpoints[:, axis] = offset
                screen, _ = self.project(
                    endpoints, angle, width=width, height=height, center=center
                )
                d.line([tuple(p) for p in screen], fill="#223541", width=1)
        if snapshot is not None:
            coords = self.snapshots[f"keys_{snapshot:03d}"] + 0.5
            colors = self.palette[self.snapshots[f"ids_{snapshot:03d}"]]
        else:
            coords = xyz[indices]
            if mode == "geometry":
                z = (coords[:, 2] - low[2]) / (high[2] - low[2])
                colors = np.clip(np.array([90, 150, 172])[None, :] + z[:, None] * 100, 0, 255)
            elif mode == "reference":
                gt = self.gt[self.indices]
                colors = self.palette[np.minimum(gt, 6)].copy()
                colors[gt == 255] = (85, 85, 85)
            else:
                colors = self.palette[labels[indices]]
        screen, depth = self.project(coords, angle, width=width, height=height, center=center)
        screen = np.rint(screen).astype(np.int32)
        valid = (screen[:, 0] > 1) & (screen[:, 0] < width - 2)
        valid &= (screen[:, 1] > 1) & (screen[:, 1] < height - 2)
        xy, depth, colors = screen[valid], depth[valid], colors[valid].astype(np.uint8)
        # Z-buffer: closest actual point wins at each pixel. No interpolated class colours.
        pixels = xy[:, 1] * width + xy[:, 0]
        order = np.lexsort((depth, pixels))
        pixels = pixels[order]
        chosen = np.r_[True, pixels[1:] != pixels[:-1]]
        xy, colors = xy[order][chosen], colors[order][chosen]
        arr = np.array(view)
        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            arr[xy[:, 1] + dy, xy[:, 0] + dx] = colors
        return Image.fromarray(arr)

    def legend(self, image, y=1100):
        for i, name in enumerate(NAMES):
            x = 54 + (i % 3) * 326
            yy = y + (i // 3) * 43
            ImageDraw.Draw(image).ellipse((x, yy + 8, x + 14, yy + 22), fill=COLORS[i])
            self.text(image, (x + 24, yy), name, 25)

    def frame(self, frame):
        seconds = frame / FPS
        if seconds < 5:
            key = "geometry"
            if key not in self.cache:
                im = self.base(
                    "01 / GENUINE AIRBORNE LiDAR",
                    "A real scene.\n400,654 source points.",
                    "IGN FRACTAL · scene C · embedded EPSG:2154",
                )
                im.paste(self.cloud_view("geometry"), (54, 414))
                self.text(
                    im, (54, 1110), "Original geometry · height shading · 10 m viewing grid", 27
                )
                self.text(
                    im,
                    (54, 1160),
                    "Source-record order retained. No fabricated flight path.",
                    27,
                    MUTED,
                )
                self.cache[key] = im
            image = self.cache[key].copy()
        elif seconds < 10:
            if "prediction" not in self.cache:
                im = self.base(
                    "02 / ACTUAL TRAINED-MODEL OUTPUT",
                    "Same geometry.\nSeven semantic channels.",
                    "Frozen 62,103-parameter 3D U-Net · native TensorRT",
                )
                im.paste(self.cloud_view(), (54, 414))
                self.legend(im)
                self.cache["prediction"] = im
            image = self.cache["prediction"].copy()
        elif seconds < 20:
            secondary = seconds >= 15 and self.secondary is not None
            image = self.base(
                "03 / A DIFFERENT VIEW",
                "One frozen model.\nDifferent real geometry."
                if secondary
                else "Native neural inference.\nOriginal point identities.",
                "Another held-out scene · 85,222 points · illustrative camera"
                if secondary
                else "Illustrative camera fly-through · recorded airborne scene",
            )
            angle = -0.85 + (seconds - 10) * 0.075
            image.paste(self.cloud_view(angle=angle, secondary=secondary), (54, 414))
            self.legend(image)
        elif seconds < 30:
            snapshot = min(
                self.proof["frames"] - 1, int((seconds - 20) / 10 * self.proof["frames"])
            )
            if ("map", snapshot) not in self.cache:
                count = len(self.snapshots[f"ids_{snapshot:03d}"])
                im = self.base(
                    "04 / RECORDED NATIVE MAP DELTAS",
                    "Observations become\na bounded semantic map.",
                    "Software replay · 1 m cells · capacity 65,536 · no TTL",
                )
                im.paste(self.cloud_view(snapshot=snapshot), (54, 414))
                self.text(
                    im,
                    (54, 1100),
                    f"Frame {snapshot + 1:02d} / {self.proof['frames']}"
                    f"   ·   {count:,} resident voxels",
                    34,
                )
                self.text(
                    im,
                    (54, 1160),
                    "Verified native deltas. Playback slowed for inspection.",
                    27,
                    MUTED,
                )
                self.text(
                    im,
                    (54, 1204),
                    "Map state is retained; it is not a fresh-classification rate.",
                    25,
                    MUTED,
                )
                self.cache[("map", snapshot)] = im
            image = self.cache[("map", snapshot)].copy()
        elif seconds < 36:
            if "limits" not in self.cache:
                im = self.base(
                    "05 / QUALITY, WITHOUT THE POLISH",
                    "Useful. Imperfect.\n48.756% seven-class mIoU.",
                    "Separate frozen 256-tile evaluation · 24,902,618 points",
                )
                im.paste(self.cloud_view("reference", width=474, height=460), (54, 475))
                im.paste(self.cloud_view(width=474, height=460), (552, 475))
                self.text(im, (54, 419), "IGN reference labels", 30)
                self.text(im, (552, 419), "Native predictions", 30)
                self.legend(im, 1000)
                n = self.proof["actual_disagreements"]
                self.text(
                    im, (54, 1160), f"This replay: {n:,} point-label disagreements with IGN.", 27
                )
                self.text(
                    im,
                    (54, 1204),
                    "Same camera / same points. Reference labels never enter inference.",
                    25,
                    MUTED,
                )
                self.cache["limits"] = im
            image = self.cache["limits"].copy()
        elif seconds < 44:
            if "metrics" not in self.cache:
                im = self.base(
                    "06 / BOUNDED RTX 4090 EVIDENCE",
                    "Fresh classifications.\nComplete map observations.",
                    "External native TCP replay · 16,384-point windows",
                )
                d = ImageDraw.Draw(im)
                for y, title, value, bar in [
                    (435, "Scene A · stable", "2.0M points/s", 920),
                    (605, "Dense C / full churn", "1.5M points/s", 690),
                ]:
                    self.text(im, (54, y), title, 30, MUTED)
                    self.text(im, (54, y + 42), value, 50)
                    d.rounded_rectangle((54, y + 110, 54 + bar, y + 124), radius=5, fill=TEAL)
                self.text(im, (54, 800), "Each: two accepted 60-second runs.", 30)
                self.text(im, (54, 866), "10 min full churn · approximately 1M points/s", 34, TEAL)
                self.text(
                    im, (54, 928), "600,014,848 fresh points · 100% coverage · zero drops", 27
                )
                self.text(im, (54, 980), "39.85 ms p99 replay-to-validated-output latency", 30)
                self.text(
                    im,
                    (54, 1072),
                    "2M dense/churn transport trials: FAIL, bounded output queue.",
                    25,
                    MUTED,
                )
                self.text(
                    im,
                    (54, 1130),
                    "Incoming pacing, fresh predictions and map deltas differ.",
                    25,
                    MUTED,
                )
                self.text(
                    im,
                    (54, 1180),
                    "These are small-model / scene-specific replay measurements.",
                    25,
                    MUTED,
                )
                self.cache["metrics"] = im
            image = self.cache["metrics"].copy()
        elif seconds < 48:
            if "qgis" not in self.cache:
                im = self.base(
                    "07 / INDEPENDENT GEOSPATIAL INSPECTION",
                    "Open in QGIS.\nTrace every source point.",
                    "Genuine QGIS map export · verified COPC · EPSG:2154",
                )
                screenshot = Image.open(self.qgis_image).convert("RGB")
                screenshot.thumbnail((972, 660), Image.Resampling.LANCZOS)
                im.paste(
                    screenshot, ((W - screenshot.width) // 2, 430 + (660 - screenshot.height) // 2)
                )
                self.text(
                    im,
                    (54, 1120),
                    "400,654 source indices and semantic IDs independently checked.",
                    25,
                )
                self.text(
                    im,
                    (54, 1170),
                    "Optional atomic file export stays outside native inference.",
                    27,
                    MUTED,
                )
                self.cache["qgis"] = im
            image = self.cache["qgis"].copy()
        else:
            if "end" not in self.cache:
                im = self.base(
                    "08 / REPRODUCIBLE ENGINEERING",
                    "Native. Bounded.\nReady for owner review.",
                    "Streaming R2 candidate · unpublished",
                )
                im.paste(self.cloud_view(width=972, height=410), (54, 440))
                self.text(im, (54, 865), "github.com/Juurikko/", 42, TEAL)
                self.text(im, (54, 920), "synrex-jetson-lidar-edge", 42, TEAL)
                self.text(im, (54, 1000), "MIT native source · public R4 foundation", 30)
                self.text(
                    im,
                    (54, 1050),
                    "IGN/FRACTAL data: Etalab 2.0 · model notices separate",
                    25,
                    MUTED,
                )
                self.text(
                    im,
                    (54, 1110),
                    "Software replay. Physical sensor / Orin / Thor: NOT_RUN.",
                    25,
                    MUTED,
                )
                self.text(
                    im,
                    (54, 1165),
                    "Tuomas Pietilä · engineering with OpenAI Codex assistance",
                    25,
                    MUTED,
                )
                self.cache["end"] = im
            image = self.cache["end"].copy()
        d = ImageDraw.Draw(image)
        d.line((54, 1281, 1026, 1281), fill="#294150", width=3)
        d.line((54, 1281, 54 + 972 * seconds / SECONDS, 1281), fill=TEAL, width=3)
        self.text(
            image,
            (54, 1295),
            "IGN FRACTAL · source updated 2025-04-05 · Etalab 2.0",
            22,
            MUTED,
        )
        return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "recording", "qgis-image", "out", "font"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--second-scene", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    renderer = Renderer(a.source, a.recording, a.qgis_image, a.font, a.second_scene)
    times = [2, 7, 15, 24, 33, 39, 46, 51]
    contact = Image.new("RGB", (W * 4 // 3, H * 2 // 3), BG)
    for i, t in enumerate(times):
        frame = renderer.frame(t * FPS)
        frame.save(a.out / f"chapter-{t:02d}.png")
        thumb = frame.resize((W // 3, H // 3), Image.Resampling.LANCZOS)
        contact.paste(thumb, ((i % 4) * (W // 3), (i // 4) * (H // 3)))
    contact.save(a.out / "contact-sheet.png")
    renderer.frame(0).save(a.out / "opening-frame.png")
    renderer.frame(7 * FPS).save(a.out / "thumbnail.png")
    if a.preview:
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(a.out / "synrex-streaming-r2-review.mp4"),
    ]
    with (a.out / "encode.log").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for i in range(FPS * SECONDS):
                process.stdin.write(renderer.frame(i).tobytes())
                if i % (FPS * 5) == 0:
                    print(f"rendered {i / FPS:.0f}/{SECONDS}s", flush=True)
        finally:
            process.stdin.close()
        assert process.wait() == 0, "encoder failed"
    video = a.out / "synrex-streaming-r2-review.mp4"
    with (a.out / "decode.log").open("w") as log:
        decoded = subprocess.run(
            [ffmpeg, "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"], stderr=log
        )
        assert decoded.returncode == 0
    # Independently decode every frame and count bytes; do not trust encoder completion.
    reader = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    count = 0
    while block := reader.stdout.read(1024 * 1024):
        count += len(block)
    assert reader.wait() == 0
    assert count == W * H * 3 * FPS * SECONDS, "decode/frame count mismatch"
    receipt = dict(
        schema="synrex.streaming_r2.media.v1",
        status="PASS",
        width=W,
        height=H,
        fps=FPS,
        seconds=SECONDS,
        decoded_frames=FPS * SECONDS,
        audio="None",
        text_bounds="PASS: all rendered text boxes within portrait safe area",
        minimum_text_pixels=22,
        all_unique_text_blocks=len(renderer.checked_text),
        source_sha256=sha(a.source),
        secondary_source=renderer.secondary_provenance,
        verification_sha256=sha(a.recording / "verification.json"),
        qgis_image_sha256=sha(a.qgis_image),
        renderer_sha256=sha(Path(__file__)),
        font_name=a.font.name,
        font_sha256=sha(a.font),
        ffmpeg_version=imageio_ffmpeg.get_ffmpeg_version(),
        viewing_decimation="Every third source point; all source records freshly classified",
        camera="Illustrative orbit only; no claimed real flight/physical capture",
        map_frames="25 independently verified recorded native-delta states; slowed playback",
        scene_quality="Actual reference/prediction disagreements; not GT-coloured predictions",
        files={
            f.name: dict(sha256=sha(f), bytes=f.stat().st_size)
            for f in a.out.iterdir()
            if f.is_file()
        },
    )
    (a.out / "media-qa.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
