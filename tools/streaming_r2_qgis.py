# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""QGIS desktop --code worker, isolated profile; verify COPC then capture actual UI.

SYNREX_QGIS_ROOT selects a disposable COPY of the verified native export bundle.
Never opens the problematic direct LAZ importer. No inference or recolouring.
"""

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from qgis.core import Qgis, QgsApplication, QgsDoubleRange, QgsGeometry, QgsProject
from qgis.PyQt.QtCore import QTimer
from qgis.utils import iface

ROOT = Path(os.environ["SYNREX_QGIS_ROOT"])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    result = dict(status="FAIL", application="Actual QGIS desktop; COPC provider", files=[])
    try:
        manifest = json.loads((ROOT / "export-receipt.json").read_text())
        for item in manifest["files"]:
            assert sha(ROOT / item["path"]) == item["sha256"]
        project = QgsProject.instance()
        assert project.read(str(ROOT / "synrex-fractal.qgz"))
        iface.mainWindow().resize(1600, 1000)
        iface.messageBar().clearWidgets()
        rows = json.loads((ROOT / "export-inputs.json").read_text())["files"]
        for row in rows:
            stem = Path(row["source"]).stem
            (layer,) = project.mapLayersByName(stem)
            assert layer.isValid() and layer.pointCount() == row["points"]
            assert layer.crs().authid() == "EPSG:2154"
            assert layer.renderer().attribute() == "Classification"
            codes = [1, 2, 200, 6, 9, 17, 201]
            assert [c.value() for c in layer.renderer().categories()] == codes
            ids = json.loads((ROOT / "data/semantics" / (stem + "-semantics.json")).read_text())[
                "semantic_ids"
            ]
            bounds = layer.extent()
            bounds.grow(0.01)
            records = layer.dataProvider().identify(
                0.00001, QgsGeometry.fromRect(bounds), QgsDoubleRange(), len(ids) + 1
            )
            seen = set()
            for p in records:
                i = int(p["SynRexSourceIndex"])
                assert i not in seen and 0 <= i < len(ids)
                seen.add(i)
                assert int(p["SynRexSemanticID"]) == ids[i]
                assert int(p["Classification"]) == codes[ids[i]]
            assert len(seen) == len(ids)
            canvas = iface.mapCanvas()
            canvas.setLayers([layer])
            canvas.setDestinationCrs(project.crs())
            bounds = layer.extent()
            bounds.scale(1.08)
            canvas.setExtent(bounds)
            iface.setActiveLayer(layer)
            canvas.refresh()
            start = time.monotonic()
            while canvas.isDrawing() or time.monotonic() - start < 3:
                if time.monotonic() - start > 60:
                    raise TimeoutError("QGIS render deadline")
                QgsApplication.processEvents()
                time.sleep(0.03)
            output = ROOT / (stem + "-streaming-r2-qgis.png")
            assert iface.mainWindow().grab().save(str(output))
            result["files"].append(
                dict(
                    source=row["source"],
                    points=len(ids),
                    coverage=1.0,
                    crs=layer.crs().authid(),
                    screenshot=output.name,
                    screenshot_sha256=sha(output),
                )
            )
        result.update(
            status="PASS",
            qgis_version=Qgis.QGIS_VERSION,
            project_sha256=sha(ROOT / "synrex-fractal.qgz"),
            actual_source_indices_and_semantics_verified=True,
            three_d="NOT_RUN: video uses separately verified scripted 3D rasterization",
        )
    except Exception:
        result["error"] = traceback.format_exc()
    (ROOT / "streaming-r2-desktop.json").write_text(json.dumps(result, indent=2) + "\n")
    project = QgsProject.instance()
    project.setDirty(False)
    QTimer.singleShot(500, iface.mainWindow().close)


QTimer.singleShot(2000, run)
