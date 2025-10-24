::free memory before the run
powershell -command "[System.GC]::Collect();"

:: to batch run, do not allow the system to sleep. recommend powertoy awake module
call thermal_stitch_only.bat D:\DATA\Milyanggang_thermal_images\Milyanggang_thermal_image_20250730\SITE01\ D:\DATA\Milyanggang_thermal_images\Milyanggang_thermal_image_20250730\SITE01\Output\
powershell.exe -Command "Start-Sleep -Seconds 10"
call thermal_stitch_only.bat D:\DATA\Milyanggang_thermal_images\Milyanggang_thermal_image_20250730\SITE02\ D:\DATA\Milyanggang_thermal_images\Milyanggang_thermal_image_20250730\SITE02\Output\
powershell.exe -Command "Start-Sleep -Seconds 10"