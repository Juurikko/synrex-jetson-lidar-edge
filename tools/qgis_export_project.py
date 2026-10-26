# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""QGIS-enabled Python worker for export_qgis.py; creates and reopens a real project."""

import hashlib
import json
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsPointCloudLayer,
    QgsProject,
    QgsReferencedRectangle,
)


def main():
    root = Path(__file__).resolve().parent
    inputs = json.loads((root / "export-inputs.json").read_text())
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    project.setTitle("SynRex FRACTAL — frozen native U-Net FP16 predictions")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    codes = [
        c["output_las_code"]
        for c in json.loads((root / "semantic_classes.json").read_text())["classes"]
    ]
    records = []
    for row in inputs["files"]:
        layer = QgsPointCloudLayer(
            str(root / row["view_filename"]), Path(row["source"]).stem, "copc"
        )
        if not layer.isValid() or layer.pointCount() != row["points"]:
            raise ValueError("invalid COPC provider/point count: " + row["source"])
        source = row["source"]
        # The conversion verifier already checks point values; QGIS supplies actual CRS parsing.
        crs_assigned = not layer.crs().isValid()
        if row.get("original_wkt"):
            reference_crs = QgsCoordinateReferenceSystem()
            if not reference_crs.createFromWkt(row["original_wkt"]) or reference_crs != layer.crs():
                raise ValueError("original CRS changed during conversion")
        elif layer.crs().isValid():
            raise ValueError("conversion introduced an undeclared CRS")
        if crs_assigned:
            fallback = inputs.get("fallback_crs")
            if not fallback or not QgsCoordinateReferenceSystem(fallback).isValid():
                raise ValueError("missing source CRS; specify a valid --fallback-crs explicitly")
            layer.setCrs(QgsCoordinateReferenceSystem(fallback))
        message, okay = layer.loadNamedStyle(str(root / "fractal-classification-r3.qml"))
        renderer = layer.renderer()
        if not okay or renderer.type() != "classified" or renderer.attribute() != "Classification":
            raise ValueError("invalid classification style: " + message)
        if [c.value() for c in renderer.categories()] != codes:
            raise ValueError("classification taxonomy mismatch")
        project.addMapLayer(layer)
        if not records:
            project.setCrs(layer.crs())
            project.viewSettings().setDefaultViewExtent(
                QgsReferencedRectangle(layer.extent(), layer.crs())
            )
        records.append(
            dict(
                source=source,
                points=layer.pointCount(),
                crs=layer.crs().authid(),
                explicit_viewing_crs_assignment=crs_assigned,
            )
        )
    filename = root / "synrex-fractal.qgz"
    project.setFileName(str(filename))
    if not project.write() or not project.read(str(filename)):
        raise ValueError("project save/reopen failed")
    if len(project.mapLayers()) != len(records) or not all(
        item.isValid() and item.renderer().type() == "classified"
        for item in project.mapLayers().values()
    ):
        raise ValueError("reopened project layers invalid")
    (root / "qgis-project.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                qgis_version=Qgis.QGIS_VERSION,
                files=records,
                project_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                scope="Real QGIS providers/save/reopen; desktop rendering qualified separately",
            ),
            indent=2,
        )
        + "\n"
    )
    project.clear()
    layer = None
    renderer = None
    app.exitQgis()


if __name__ == "__main__":
    main()
