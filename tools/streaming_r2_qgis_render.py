# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""QGIS-enabled Python: verify every COPC source index, then export a real map image.

SYNREX_QGIS_ROOT must reference a disposable copy of a verified export package.
This is an actual QGIS provider/renderer export, not a desktop screenshot or 3D view.
"""

import hashlib
import json
import os
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsDoubleRange,
    QgsGeometry,
    QgsMapRendererParallelJob,
    QgsMapSettings,
    QgsProject,
)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor


def main():
    root = Path(os.environ["SYNREX_QGIS_ROOT"])
    (root / "render-started.txt").write_text("QGIS Python entered\n")
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    assert project.read(str(root / "synrex-fractal.qgz"))
    inputs = json.loads((root / "export-inputs.json").read_text())
    result = dict(
        status="PASS",
        qgis_version=Qgis.QGIS_VERSION,
        files=[],
        kind="Actual QGIS map image export; not a desktop screenshot or 3D view",
    )
    for row in inputs["files"]:
        stem = Path(row["source"]).stem
        (layer,) = project.mapLayersByName(stem)
        assert layer.isValid() and layer.pointCount() == row["points"]
        assert layer.crs().authid() == "EPSG:2154"
        ids = json.loads((root / "data/semantics" / (stem + "-semantics.json")).read_text())[
            "semantic_ids"
        ]
        codes = [1, 2, 200, 6, 9, 17, 201]
        assert [c.value() for c in layer.renderer().categories()] == codes
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
        settings = QgsMapSettings()
        settings.setLayers([layer])
        settings.setDestinationCrs(layer.crs())
        settings.setOutputSize(QSize(1600, 1200))
        settings.setBackgroundColor(QColor("#101f2d"))
        settings.setExtent(bounds)
        job = QgsMapRendererParallelJob(settings)
        job.start()
        job.waitForFinished()
        assert not job.errors(), str(job.errors())
        image = root / (stem + "-qgis-map.png")
        assert job.renderedImage().save(str(image))
        result["files"].append(
            dict(
                source=row["source"],
                source_indices=len(seen),
                coverage=1.0,
                crs=layer.crs().authid(),
                image=image.name,
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            )
        )
    (root / "streaming-r2-render.json").write_text(json.dumps(result, indent=2) + "\n")
    project.clear()
    layer = None
    job = None
    app.exitQgis()


if __name__ == "__main__":
    main()
