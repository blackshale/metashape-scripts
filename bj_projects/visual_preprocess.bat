::free memory before run
powershell -command "[System.GC]::Collect();"

echo Input Image Folder = %1
echo Resolution = %2

:: run python program
"C:\Users\cran003\miniconda3\condabin\conda.bat" run --no-capture-output -n metashape-scripts python D:\github\metashape-scripts\bj_projects\downscale_image.py "%1W" %1W_"%2" "--size" %2

goto :eof

