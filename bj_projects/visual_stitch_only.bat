


::free memory before run
powershell -command "[System.GC]::Collect();"

echo Site folder = %1
echo Resolution = %2

:: the metashpa python does not allow to add packages...
"C:\Program Files\Agisoft\Metashape Pro\metashape.exe" -r D:\github\metashape-scripts\bj_projects\metashape_workflow.py  %1W_"%2" %1Output_visual_"%2" "-k 10000" "-t 20000"


:: you have run additional python programs for post processing separately.
"C:\Users\cran003\miniconda3\condabin\conda.bat" run -n metashape-scripts python D:\github\metashape-scripts\bj_projects\strip_alpha_to_nodata.py %1Output_visual_"%2"




goto :eof

