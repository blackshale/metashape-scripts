::free memory before the run
powershell -command "[System.GC]::Collect();"

echo "batch_xxx.bat [base folder] site01(subfolder name) site02 site03"


:: to batch run, do not allow the system to sleep. recommend powertoy awake module
@echo off
REM ================================================
REM Batch script to run thermal_stitch_only.bat for multiple sites
REM Prevent system sleep: use PowerToys Awake or keep system active manually
REM ================================================

setlocal enabledelayedexpansion

REM ==== CLI args: %1 = BASE, %2.. = sites ====
if not "%~1"=="" set "BASE=%~1"

set "SITES="
:__collect_sites
shift
if "%~1"=="" goto __sites_done
set "SITES=%SITES% %~1"
goto __collect_sites
:__sites_done

for %%S in (%SITES%) do (
    echo =====================================================
    echo [%%S] Downscaling images ...
    echo =====================================================

    set "INPUT=%BASE%%%S\"

    if not exist "!INPUT!" (
        echo [ERROR] Input folder not found: !INPUT!
        echo Skipping %%S...
        echo
    )

    REM preprocess for visual photo downscaling, resolution down 8000x6000 --> 1600x1200
    call visual_preprocess.bat !INPUT!

    echo [INFO] Waiting 10 seconds before next job...
    powershell.exe -Command "Start-Sleep -Seconds 10"

)

echo =====================================================
echo All sites processed. You may close this window.
echo =====================================================

endlocal
pause
