# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""QGIS --code desktop validation of an already generated export/project."""

import collections
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from qgis.core import QgsApplication, QgsDoubleRange, QgsGeometry, QgsProject
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import QDockWidget
from qgis.utils import iface

ROOT = Path(os.environ["SYNREX_QGIS_ROOT"])


def qualify():
    result = dict(status="FAIL", files=[])
    try:
        manifest = json.loads((ROOT / "export-receipt.json").read_text())
        for row in manifest["files"]:
            if hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("export manifest mismatch")
        project = QgsProject.instance()
        if not project.read(str(ROOT / "synrex-fractal.qgz")):
            raise ValueError("desktop project open failed")
        inputs = json.loads((ROOT / "export-inputs.json").read_text())
        codes = [
            c["output_las_code"]
            for c in json.loads((ROOT / "semantic_classes.json").read_text())["classes"]
        ]
        observed = set()
        iface.mainWindow().resize(1600, 1000)
        iface.messageBar().clearWidgets()
        for dock in iface.mainWindow().findChildren(QDockWidget):
            if "browser" in dock.objectName().lower():
                dock.hide()
            if dock.windowTitle() == "Layers":
                iface.mainWindow().resizeDocks([dock], [440], Qt.Horizontal)
        for row in inputs["files"]:
            stem = Path(row["source"]).stem
            (layer,) = project.mapLayersByName(stem)
            if not layer.isValid() or layer.pointCount() != row["points"]:
                raise ValueError("desktop layer invalid")
            if [c.value() for c in layer.renderer().categories()] != codes:
                raise ValueError("desktop taxonomy mismatch")
            bounds = layer.extent()
            bounds.grow(0.01)
            points = layer.dataProvider().identify(
                0.00001, QgsGeometry.fromRect(bounds), QgsDoubleRange(), row["points"] + 1
            )
            ids = json.loads((ROOT / "data/semantics" / (stem + "-semantics.json")).read_text())[
                "semantic_ids"
            ]
            indices = set()
            counts = collections.Counter()
            for point in points:
                index = int(point["SynRexSourceIndex"])
                if not 0 <= index < len(ids) or index in indices:
                    raise ValueError("source identity not a permutation")
                indices.add(index)
                semantic = int(point["SynRexSemanticID"])
                code = int(point["Classification"])
                if semantic != ids[index] or code != codes[semantic]:
                    raise ValueError("desktop semantic mismatch")
                counts[semantic] += 1
            if len(indices) != len(ids):
                raise ValueError("desktop point coverage mismatch")
            observed.update(counts)
            canvas = iface.mapCanvas()
            canvas.setLayers([layer])
            canvas.setDestinationCrs(project.crs())
            extent = layer.extent()
            extent.scale(1.08)
            canvas.setExtent(extent)
            iface.setActiveLayer(layer)
            canvas.refresh()
            start = time.monotonic()
            while canvas.isDrawing() or time.monotonic() - start < 1:
                if time.monotonic() - start > 60:
                    raise TimeoutError("render timeout")
                QgsApplication.processEvents()
                time.sleep(0.03)
            screenshot = ROOT / (stem + "-qgis.png")
            if not iface.mainWindow().grab().save(str(screenshot)):
                raise ValueError("screenshot failed")
            result["files"].append(
                dict(
                    source=row["source"],
                    points=len(ids),
                    semantic_counts=dict(counts),
                    crs=layer.crs().authid(),
                    screenshot=screenshot.name,
                    screenshot_sha256=hashlib.sha256(screenshot.read_bytes()).hexdigest(),
                )
            )
        if observed != set(range(7)):
            raise ValueError("showcase does not observe all seven semantic classes")
        result.update(
            status="PASS",
            all_seven_semantics_observed=True,
            exact_source_index_and_semantic_coverage=True,
            actual_exported_project_loaded=True,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    (ROOT / "desktop-qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    QgsProject.instance().setDirty(False)
    QTimer.singleShot(500, iface.mainWindow().close)


QTimer.singleShot(2000, qualify)
