#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agisoft Metashape 1.8.5–compatible workflow (thermal_workflow2.py)

- Single-pass matching with configurable keypoint_limit & tiepoint_limit
- Highest accuracy (downscale=0), Generic + Reference preselection (ReferencePreselectionSource)
- Align cameras with only camera fitting (adaptive_fitting=False)
- Reset region and enlarge to ensure full coverage
- Build Ultra High depth maps -> Dense Cloud -> DEM (geographic CRS; interpolation enabled)
- Build Orthomosaic from DEM with hole filling, ghosting filter, and seamline refinement
- Export Orthomosaic to GeoTIFF with max dimension cap and TFW world file (orthomosaic nodata via alpha)

Usage (GUI-less):
    metashape.sh -r thermal_workflow2.py <image_folder> <output_folder> [-k KEYPOINT] [-t TIEPOINT] [-m EXPORT_MAX_DIM]
"""

import os
import sys
import argparse
import Metashape

# -----------------------------
# Configuration (defaults)
# -----------------------------
MATCH_DOWNSCALE = 0  # 0 = Highest (align accuracy)
REF_PRESEL_MODE = Metashape.ReferencePreselectionMode.ReferencePreselectionSource

# Depth maps (Ultra High for 1.8.5 -> downscale=1) and mild filtering
DEPTHMAPS_DOWNSCALE = 1
DEPTHMAPS_FILTER = Metashape.MildFiltering

TARGET_EPSG = os.environ.get("MS_TARGET_EPSG", "EPSG::4326")
ORTHO_PIXEL_SIZE = float(os.environ.get("MS_ORTHO_PIXEL_SIZE", "0"))
REGION_SCALE_FACTOR = float(os.environ.get("MS_REGION_SCALE_FACTOR", "1.2"))
TRY_QUICK_LAYOUT = os.environ.get("MS_TRY_QUICK_LAYOUT", "1") == "1"


def _find_images(folder):
    exts = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
    return [
        os.path.join(folder, n)
        for n in sorted(os.listdir(folder))
        if os.path.splitext(n)[1].lower() in exts
    ]


def _save(doc):
    doc.save(doc.path, archive=True)


def _ensure_dirs(path):
    os.makedirs(path, exist_ok=True)


def _reset_and_expand_region(chunk, scale_factor=1.2):
    """
    Reset region and enlarge to reduce clipping risk.
    """
    chunk.resetRegion()
    region = chunk.region
    size = region.size
    region.size = Metashape.Vector([size.x * scale_factor, size.y * scale_factor, size.z * scale_factor])
    chunk.region = region


def _apply_vertical_quick_layout_if_available(chunk):
    """
    Optional helper (if you provide quick_layout.py with apply_vertical_camera_alignment(chunk)).
    """
    try:
        import quick_layout
        if hasattr(quick_layout, "apply_vertical_camera_alignment"):
            quick_layout.apply_vertical_camera_alignment(chunk)
            print("[quick_layout] Applied vertical camera alignment")
    except Exception as e:
        print(f"[quick_layout] Skipped: {e}")


def _set_geographic_crs(chunk, epsg):
    chunk.crs = Metashape.CoordinateSystem(epsg)
    chunk.updateTransform()


def _match_and_align(chunk, keypoint_limit, tiepoint_limit):
    """
    Single-pass matching + alignment for Metashape 1.8.5.
    """
    print("== Match Photos (single pass) ==")
    chunk.matchPhotos(
        downscale=MATCH_DOWNSCALE,
        generic_preselection=True,
        reference_preselection=True,
        reference_preselection_mode=REF_PRESEL_MODE,
        keypoint_limit=int(keypoint_limit),
        tiepoint_limit=int(tiepoint_limit),
        keep_keypoints=False,
        reset_matches=True,
    )
    # Align cameras with "only camera fitting" (no adaptive distortion fitting)
    chunk.alignCameras(adaptive_fitting=False)
    # Optional: triangulate to solidify tie point cloud
    try:
        chunk.triangulatePoints(max_error=10, min_image=2)
    except Exception as e:
        print(f"[warn] triangulatePoints skipped: {e}")


def _build_dem_orthomosaic(chunk):
    """
    Depth maps (Ultra High) -> Dense Cloud -> DEM (interpolated) -> Orthomosaic (from DEM).
    """
    # Depth maps at Ultra High for 1.8.5
    chunk.buildDepthMaps(
        downscale=DEPTHMAPS_DOWNSCALE,
        filter_mode=DEPTHMAPS_FILTER,
    )

    # Dense point cloud from depth maps; keep depth maps for reuse
    chunk.buildDenseCloud(keep_depth=True)

    # DEM from dense cloud with interpolation enabled (best terrain continuity)
    chunk.buildDem(
        source_data=Metashape.DenseCloudData,
        interpolation=Metashape.EnabledInterpolation,
    )

    # Orthomosaic from DEM with quality options
    chunk.buildOrthomosaic(
        surface_data=Metashape.ElevationData,
        fill_holes=True,
        ghosting_filter=True,
        refine_seamlines=True,
        cull_faces=False,
        resolution=ORTHO_PIXEL_SIZE,  # 0 -> auto
    )


def _export_orthomosaic_with_caps(chunk, out_folder, max_dim):
    """
    Export orthomosaic to GeoTIFF with a max dimension cap and TFW world file.
    Nodata for orthomosaic is represented by alpha (DEM nodata is numeric).
    """
    _ensure_dirs(out_folder)

    ortho = chunk.orthomosaic
    width = ortho.width
    height = ortho.height

    scale = 1.0
    if max(width, height) > max_dim:
        scale = max_dim / float(max(width, height))
    export_w = max(1, int(round(width * scale)))
    export_h = max(1, int(round(height * scale)))

    out_tif = os.path.join(out_folder, "orthomosaic.tif")
    chunk.exportRaster(
        path=out_tif,
        source_data=Metashape.OrthomosaicData,
        save_world=True,   # write TFW
        save_alpha=True,   # carry nodata as alpha
        width=export_w,
        height=export_h,
    )
    print(f"[export] Orthomosaic -> {out_tif}  ({export_w} x {export_h}, alpha nodata)")  # fixed missing )


def _parse_args():
    p = argparse.ArgumentParser(description="Metashape 1.8.5 workflow (single-pass matching with CLI knobs).")
    p.add_argument("image_folder", help="Folder with input images")
    p.add_argument("output_folder", help="Folder for project & exports")
    p.add_argument("-k", "--keypoint-limit", type=int,
                   default=int(os.environ.get("MS_KEYPOINT_LIMIT", "10000")),
                   help="Keypoint limit for matchPhotos (default: env MS_KEYPOINT_LIMIT or 20000)")
    p.add_argument("-t", "--tiepoint-limit", type=int,
                   default=int(os.environ.get("MS_TIEPOINT_LIMIT", "20000")),
                   help="Tie point (max point) limit for matchPhotos (default: env MS_TIEPOINT_LIMIT or 20000)")
    p.add_argument("-m", "--export-max-dim", type=int,
                   default=int(os.environ.get("MS_EXPORT_MAX_DIM", "4096")),
                   help="Max dimension (pixels) for orthomosaic export (default: env MS_EXPORT_MAX_DIM or 4096)")
    return p.parse_args()


def main():
    args = _parse_args()

    image_folder = args.image_folder
    output_folder = args.output_folder
    _ensure_dirs(output_folder)

    photos = _find_images(image_folder)
    if not photos:
        raise RuntimeError(f"No images found in {image_folder}")

    # Prepare project
    doc = Metashape.Document()
    project_path = os.path.join(output_folder, "project.psx")
    doc.save(project_path, archive=True)

    chunk = doc.addChunk()
    chunk.addPhotos(photos)
    _save(doc)
    print(f"{len(chunk.cameras)} images loaded")

    # Optional vertical layout helper
    if TRY_QUICK_LAYOUT:
        _apply_vertical_quick_layout_if_available(chunk)

    # Matching + alignment (single pass with CLI-configurable limits)
    _match_and_align(chunk, args.keypoint_limit, args.tiepoint_limit)
    _save(doc)

    # Safe region
    _reset_and_expand_region(chunk, REGION_SCALE_FACTOR)
    _save(doc)

    # Geographic CRS (e.g., EPSG:4326)
    _set_geographic_crs(chunk, TARGET_EPSG)
    _save(doc)

    # Depth maps -> Dense Cloud -> DEM -> Orthomosaic
    _build_dem_orthomosaic(chunk)
    _save(doc)

    # Export processing report (if available in 1.8.5)
    try:
        chunk.exportReport(os.path.join(output_folder, "report.pdf"))
    except Exception as e:
        print(f"[warn] exportReport failed: {e}")

    # Export DEM (numeric nodata supported in DEM export)
    if chunk.elevation:
        dem_path = os.path.join(output_folder, "dem.tif")
        try:
            chunk.exportRaster(
                path=dem_path,
                source_data=Metashape.ElevationData,
                save_world=True,
                nodata_value=-32767,
            )
            print(f"[export] DEM -> {dem_path}")
        except Exception as e:
            print(f"[warn] DEM export failed: {e}")

    # Export Orthomosaic (alpha nodata) with size cap
    if chunk.orthomosaic:
        _export_orthomosaic_with_caps(chunk, output_folder, args.export_max_dim)
    else:
        print("[warn] No orthomosaic to export")

    print("Processing finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
