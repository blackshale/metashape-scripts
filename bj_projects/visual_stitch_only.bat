


::free memory before run
powershell -command "[System.GC]::Collect();"

echo "%1 = site folder"

:: the metashpa python does not allow to add packages...
"C:\Program Files\Agisoft\Metashape Pro\metashape.exe" -r D:\github\metashape-scripts\bj_projects\metashape_workflow.py  "%1W" "%1Output_visual" "-k 10000" "-t 20000"




goto :eof

