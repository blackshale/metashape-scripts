


::free memory before run
powershell -command "[System.GC]::Collect();"

echo Site folder = %1

:: the metashpa python does not allow to add packages...
"C:\Program Files\Agisoft\Metashape Pro\metashape.exe" -r D:\github\metashape-scripts\bj_projects\patch_orthomosaic.py %1

:: you have run additional python programs for post processing separately.
"C:\Users\cran003\miniconda3\condabin\conda.bat" run -n metashape-scripts python D:\github\metashape-scripts\bj_projects\strip_alpha_to_nodata.py %1


goto :eof

