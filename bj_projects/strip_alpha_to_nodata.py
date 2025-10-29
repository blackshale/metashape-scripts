#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create single-band 'orthomosaic_noalpha.tif' from 'orthomosaic.tif' with:
  - NoData = -32767 where alpha == 0
  - GeoTIFF georeferencing copied (GeoTransform, Projection, GCPs/RPCs)

Assumptions:
  band 1 = temperature, band 2 = alpha
Files are read/written in the SAME folder as this script.
"""

import sys
import os

def _strip_alpha_to_nodata(working_folder):

    import os
    import numpy as np
    from osgeo import gdal,osr

    NODATA = -32767

    SCRIPT_DIR = working_folder
    IN_PATH    = os.path.join(SCRIPT_DIR, "orthomosaic.tif")
    OUT_PATH   = os.path.join(SCRIPT_DIR, "orthomosaic_noalpha.tif")

    gdal.UseExceptions()

    # Open source
    src = gdal.Open(IN_PATH, gdal.GA_ReadOnly)
    if src is None:
        raise FileNotFoundError(f"Cannot open {IN_PATH}")

    # Expect at least 2 bands
    if src.RasterCount < 2:
        raise RuntimeError(f"Expected at least 2 bands (temp + alpha); found {src.RasterCount}")

    band_temp = src.GetRasterBand(1)
    band_alpha = src.GetRasterBand(2)

    xsize = src.RasterXSize
    ysize = src.RasterYSize
    dtype = band_temp.DataType  # keep same base type if practical

    # Choose output dtype: use Int16 if input is integer; else Float32
    if dtype in (gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_Int32, gdal.GDT_UInt32):
        out_dtype = gdal.GDT_Int16
        nodata_val = NODATA
    else:
        out_dtype = gdal.GDT_Float32
        nodata_val = float(NODATA)

    # Read in chunks (works for big rasters)
    block_x = min(1024, xsize)
    block_y = min(1024, ysize)

    # Create destination with same georeferencing
    drv = gdal.GetDriverByName("GTiff")
    # Creation options: tiled + deflate + predictor + BigTIFF if needed
    co = ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "BIGTIFF=IF_NEEDED"]
    dst = drv.Create(OUT_PATH, xsize, ysize, 1, out_dtype, options=co)
    if dst is None:
        raise RuntimeError(f"Failed to create {OUT_PATH}")

    # Copy GeoTransform / Projection
    gt = src.GetGeoTransform(can_return_null=True)
    if gt:
        dst.SetGeoTransform(gt)
    proj = src.GetProjectionRef()
    if proj:
        dst.SetProjection(proj)

    # Copy GCPs if present
    gcps = src.GetGCPs()
    if gcps:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(src.GetGCPProjection())
        dst.SetGCPs(gcps, srs.ExportToWkt())

    # Copy RPCs if present
    rpc_md = src.GetMetadata("RPC")
    if rpc_md:
        dst.SetMetadata(rpc_md, "RPC")

    out_band = dst.GetRasterBand(1)
    out_band.SetNoDataValue(nodata_val)
    out_band.SetDescription("temperature (NoData = -32767)")
    out_band.SetColorInterpretation(gdal.GCI_GrayIndex)  # single-band gray

    # Process in blocks
    for y in range(0, ysize, block_y):
        rows = min(block_y, ysize - y)
        temp = band_temp.ReadAsArray(0, y, xsize, rows).astype(np.float32, copy=False)
        alpha = band_alpha.ReadAsArray(0, y, xsize, rows)

        # Convert temp to desired dtype after masking
        mask = (alpha == 0)
        if out_dtype == gdal.GDT_Int16:
            tile = temp.astype(np.int16, copy=False)
            tile[mask] = np.int16(NODATA)
        else:
            tile = temp.astype(np.float32, copy=False)
            tile[mask] = float(NODATA)

        out_band.WriteArray(tile, xoff=0, yoff=y)

    # Build stats (optional, helps some apps)
    out_band.ComputeStatistics(False)

    # Flush & close
    out_band = None
    dst = None
    band_temp = None
    band_alpha = None
    src = None

    print(f"[ok] Wrote {OUT_PATH} with NoData={nodata_val} and original georeferencing.")
    return 0

if __name__ == "__main__" :
    if len(sys.argv) > 1:
        working_folder = sys.argv[1]
        sys.exit(_strip_alpha_to_nodata(working_folder))
else:
    sys.exit(_strip_alpha_to_nodata(os.path.dirname(os.path.abspath(__file__))))