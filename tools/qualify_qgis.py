# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Run with QGIS --code in an isolated profile over a prepared immutable package.

Requires QGIS GUI, providers and Qt; never modifies classified/source LAZ files.
"""

import collections
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsDoubleRange,
    QgsGeometry,
    QgsPointCloudLayer,
    QgsProject,
)
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import QDockWidget
from qgis.utils import iface

ROOT = Path(os.environ.get("SYNREX_QGIS_ROOT") or Path.cwd())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pump_until(predicate, timeout=180):
    start = time.monotonic()
    while not predicate():
        if time.monotonic() - start > timeout:
            raise TimeoutError("QGIS provider/render timeout")
        QgsApplication.processEvents()
        time.sleep(0.03)


def qualify():
    (ROOT / "script-started.txt").write_text("QGIS code callback entered")
    result = dict(
        status="FAIL",
        qgis_version=Qgis.QGIS_VERSION,
        application="Actual QGIS desktop executable, isolated profile, --code",
        files=[],
    )
    try:
        selection = json.loads((ROOT / "selection.json").read_text())
        immutable = json.loads((ROOT / "immutable-inputs.json").read_text())
        for item in immutable:
            assert sha(ROOT / item["path"]) == item["sha256"]
        project = QgsProject.instance()
        project.clear()
        project.setTitle("SynRex frozen FRACTAL R2 — genuine native FP16 predictions")
        project.setFilePathStorage(Qgis.FilePathType.Relative)
        iface.mainWindow().resize(1600, 1000)
        for dock in iface.mainWindow().findChildren(QDockWidget):
            if "browser" in dock.objectName().lower():
                dock.hide()
            if dock.windowTitle() == "Layers":
                iface.mainWindow().resizeDocks([dock], [440], Qt.Horizontal)
        classes = json.loads((ROOT / "semantic_classes.json").read_text())["classes"]
        codes = [c["output_las_code"] for c in classes]
        all_codes = set()
        copc_mode = bool(selection.get("copc_views"))
        layers = []
        for row in selection["files"]:
            stem = Path(row["source"]).stem
            output = (
                ROOT / row["view_filename"]
                if copc_mode
                else ROOT / "data/classified" / (stem + "-classified.laz")
            )
            layer = QgsPointCloudLayer(
                str(output), stem + " · native U-Net FP16", "copc" if copc_mode else "pdal"
            )
            assert layer.isValid(), layer.error().summary()
            provider = layer.dataProvider()
            pump_until(provider.hasValidIndex)
            assert layer.pointCount() == row["points"]
            style = ROOT / "fractal-classification-r3.qml"
            if not style.exists():
                style = ROOT / "fractal-classification.qml"
            message, okay = layer.loadNamedStyle(str(style))
            assert okay, message
            renderer = layer.renderer()
            assert renderer.type() == "classified" and renderer.attribute() == "Classification"
            assert [c.value() for c in renderer.categories()] == codes
            assert all(renderer.willRenderPoint({"Classification": code}) for code in codes)
            if copc_mode:
                reference_crs = QgsCoordinateReferenceSystem()
                assert reference_crs.createFromWkt(row["original_wkt"])
                assert reference_crs == layer.crs()
                crs_assigned = False
            else:
                options = QgsPointCloudLayer.LayerOptions()
                options.skipIndexGeneration = True
                options.skipStatisticsCalculation = True
                original = QgsPointCloudLayer(
                    str(ROOT / "data/source" / row["source"]),
                    "source CRS reference",
                    "pdal",
                    options,
                )
                assert original.isValid(), original.error().summary()
                assert original.crs().isValid() == layer.crs().isValid()
                crs_assigned = not layer.crs().isValid()
                if crs_assigned:
                    # Explicit horizontal viewer assignment; no vertical datum claim.
                    layer.setCrs(QgsCoordinateReferenceSystem("EPSG:2154"))
                    original.setCrs(QgsCoordinateReferenceSystem("EPSG:2154"))
                assert original.crs() == layer.crs()
            if not layers:
                project.setCrs(layer.crs())
            project.addMapLayer(layer)
            layers.append(layer)
            # Request leaf-resolution points via the actual QGIS provider, not laspy.
            bounds = layer.extent()
            bounds.grow(0.01)
            points = provider.identify(
                0.00001, QgsGeometry.fromRect(bounds), QgsDoubleRange(), row["points"] + 1
            )
            histogram = collections.Counter(int(p["Classification"]) for p in points)
            expected = {code: n for code, n in zip(codes, row["semantic_counts"], strict=True) if n}
            assert dict(histogram) == expected, (len(points), row["points"], histogram, expected)
            if copc_mode:
                semantics = json.loads(
                    (ROOT / "data/semantics" / (stem + "-semantics.json")).read_text()
                )["semantic_ids"]
                seen_indices = set()
                for point in points:
                    index = int(point["SynRexSourceIndex"])
                    assert 0 <= index < len(semantics) and index not in seen_indices
                    seen_indices.add(index)
                    assert int(point["SynRexSemanticID"]) == semantics[index]
                    assert int(point["Classification"]) == codes[semantics[index]]
                assert len(seen_indices) == len(semantics)
            all_codes.update(histogram)
            canvas = iface.mapCanvas()
            canvas.setDestinationCrs(project.crs())
            canvas.setLayers([layer])
            iface.setActiveLayer(layer)
            display_extent = layer.extent()
            display_extent.scale(1.08)
            canvas.setExtent(display_extent)
            canvas.refresh()
            pump_until(lambda canvas=canvas: not canvas.isDrawing())
            # Process queued paint events before taking an actual application screenshot.
            for _ in range(20):
                QgsApplication.processEvents()
                time.sleep(0.03)
            image = ROOT / (stem + "-qgis.png")
            assert iface.mainWindow().grab().save(str(image))
            result["files"].append(
                dict(
                    source=row["source"],
                    points=layer.pointCount(),
                    qgis_provider=provider.name(),
                    crs_authid=layer.crs().authid(),
                    source_crs_matches=True,
                    missing_crs_explicit_horizontal_assignment=crs_assigned,
                    qgis_classification_histogram=dict(histogram),
                    screenshot=image.name,
                    screenshot_sha256=sha(image),
                    output_sha256=sha(output),
                )
            )
            print("QGIS validated", stem, flush=True)
        if not copc_mode:
            assert all_codes == set(codes)
        project.setFileName(str(ROOT / "synrex-fractal-r2.qgz"))
        assert project.write()
        # Exercise an actual project CRS transformation independently of source storage.
        original_crs = project.crs()
        geographic = QgsCoordinateReferenceSystem("EPSG:4326")
        transform = QgsCoordinateTransform(layers[0].crs(), geographic, project)
        geo_extent = transform.transformBoundingBox(layers[0].extent())
        assert -180 <= geo_extent.xMinimum() <= 180 and -90 <= geo_extent.yMinimum() <= 90
        project.setCrs(geographic)
        iface.mapCanvas().setDestinationCrs(geographic)
        iface.mapCanvas().setLayers([layers[0]])
        iface.mapCanvas().setExtent(geo_extent)
        iface.mapCanvas().refresh()
        pump_until(lambda: not iface.mapCanvas().isDrawing())
        project.setCrs(original_crs)
        assert project.read(str(ROOT / "synrex-fractal-r2.qgz"))
        assert len(project.mapLayers()) == len(layers)
        assert all(
            layer.isValid() and layer.renderer().type() == "classified"
            for layer in project.mapLayers().values()
        )
        for item in immutable:
            assert sha(ROOT / item["path"]) == item["sha256"], "Input modified by viewer"
        result.update(
            status="PASS",
            all_seven_categories_seen=all_codes == set(codes),
            copc_source_indices_and_semantics_verified=copc_mode,
            exact_qgis_provider_histograms=True,
            project_reopen="PASS",
            crs_reprojection="PASS",
            project_crs=project.crs().authid(),
            project_sha256=sha(ROOT / "synrex-fractal-r2.qgz"),
            immutable_inputs_unchanged=True,
            selection=selection["selection"],
        )
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    result["harness_sha256"] = sha(ROOT / "qualify_qgis.py")
    (ROOT / "qgis-qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    QgsProject.instance().setDirty(False)
    # Close through QgisApp after callback-local provider/layer references release.
    QTimer.singleShot(500, iface.mainWindow().close)


QTimer.singleShot(2000, qualify)
