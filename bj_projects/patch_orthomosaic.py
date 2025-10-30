#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export orthomosaic from a Metashape 1.8.5 project (output next to .psx).

Enhancements:
 - Accepts either a full path to project.psx OR a folder containing it
 - Pixel size: automatic (native)
 - CRS: uses orthomosaic's CRS or chunk's CRS
 - Setup boundaries before export
 - World file (TFW): ON
 - Save alpha channel: ON
 - TIFF compression: LZW (no tiled, no BigTIFF, no overviews)
 - Clip to boundary shapes: OFF
"""

import os
import sys
import argparse
import Metashape


def _resolve_psx_path(path):
    """Accepts a folder or full path and returns an absolute path to project.psx"""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        psx_files = [f for f in os.listdir(path) if f.lower().endswith(".psx")]
        if not psx_files:
            raise FileNotFoundError(f"No .psx file found in folder: {path}")
        psx_path = os.path.join(path, psx_files[0])
        print(f"[info] Using project file: {psx_path}")
        return psx_path
    elif os.path.isfile(path):
        if not path.lower().endswith(".psx"):
            raise ValueError(f"Provided file is not a .psx: {path}")
        return path
    else:
        raise FileNotFoundError(f"Invalid path: {path}")


def _load_project(psx_path):
    psx_path = os.path.abspath(psx_path)
    doc = Metashape.Document()
    try:
        doc.open(psx_path)
    except Exception as e:
        try:
            Metashape.app.document.open(psx_path)
            doc = Metashape.app.document
        except Exception as e2:
            raise RuntimeError(
                f"Failed to open project (both methods): {psx_path}\n"
                f"Document.open() error: {e}\nApp.document.open() error: {e2}"
            )
    if not getattr(doc, "chunks", None):
        raise RuntimeError("Project opened but has no chunks.")
    return doc


def _active_chunk(doc):
    if doc.chunk is None:
        doc.chunk = doc.chunks[0]
    return doc.chunk


def _setup_orthomosaic_boundary(chunk):
    """Equivalent to checking 'Setup boundaries' in the GUI."""
    if not chunk.orthomosaic:
        raise RuntimeError("Chunk has no orthomosaic. Build one before exporting.")
    ortho = chunk.orthomosaic
    for meth in ("estimateBoundary", "calculateBoundary", "setupBoundary", "updateBoundary"):
        fn = getattr(ortho, meth, None)
        if callable(fn):
            print(f"[boundary] orthomosaic.{meth}()")
            fn()
            return
    print("[boundary] No boundary-estimation method found; continuing.")


def export_orthomosaic(input_path):
    psx_path = _resolve_psx_path(input_path)
    project_dir = os.path.dirname(psx_path)
    out_tif = os.path.join(project_dir, "orthomosaic.tif")

    doc = _load_project(psx_path)
    print(f"[open] {doc.path or '(in-memory document)'}")
    chunk = _active_chunk(doc)

    if not chunk.orthomosaic:
        raise RuntimeError("No orthomosaic found in the active chunk. Build one before exporting.")

    _setup_orthomosaic_boundary(chunk)

    # Compression settings
    comp = Metashape.ImageCompression()
    try:
        comp.tiff_compression = Metashape.ImageCompression.TiffCompressionLZW
    except AttributeError:
        comp.tiff_compression = "lzw"
    comp.jpeg_quality = 90
    comp.tiff_big = False
    comp.tiff_tiled = False
    if hasattr(comp, "tiff_overviews"):
        comp.tiff_overviews = False
    if hasattr(comp, "generate_tiff_overviews"):
        comp.generate_tiff_overviews = False

    # Export options
    export_kwargs = dict(
        path=out_tif,
        source_data=Metashape.OrthomosaicData,
        save_world=True,
        save_alpha=True,
        clip_to_boundary=False,
        image_compression=comp,
    )


    print("[export] Exporting orthomosaic...")
    chunk.exportRaster(**export_kwargs)
    print(f"[done] Exported orthomosaic: {out_tif}")


def _parse_args():
    p = argparse.ArgumentParser(description="Export orthomosaic (TIFF) from project.psx or its folder.")
    p.add_argument("input_path", help="Path to project.psx or a folder containing it")
    return p.parse_args()


def main():
    args = _parse_args()
    export_orthomosaic(args.input_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
