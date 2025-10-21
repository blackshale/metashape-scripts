echo "%1 = image folder, %2 = output folder"


::  path conversion (dos to linux)
:: input 
set "inppath=%~1"
set "inppath=%inppath:\=/%"
set "inppath=%inppath:D:/=/mnt/d/%"
:: output
set "outpath=%~2"
set "outpath=%outpath:\=/%"
set "outpath=%outpath:D:/=/mnt/d/%"


:: ------------------------------------
:: STITCH PICTURES
:: ------------------------------------
:: Replace paths as needed. Optional env vars shown for tuning
::set $env:MS_ORTHO_PIXEL_SIZE="0.3"         # 0 = let Metashape decide; or set e.g. 0.03 (3 cm)
::set $env:MS_EXPORT_MAX_DIM="4096"        # cap larger dimension on export
::set $env:MS_ORTHO_NODATA="-32750"        # nodata index for orthomosaic
::set $env:MS_TARGET_EPSG="EPSG::4326"   # geographic CRS (WGS84)


"C:\Program Files\Agisoft\Metashape Pro\metashape.exe" -r D:\github\metashape-scripts\bj_projects\thermal_workflow2.py  "%1T-C" %2



goto :eof

