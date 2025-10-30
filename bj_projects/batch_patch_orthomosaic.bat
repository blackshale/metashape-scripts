::free memory before the run
powershell -command "[System.GC]::Collect();"

echo "batch_xxx.bat [base folder] [output folder] site01(subfolder name) site02 site03"


:: to batch run, do not allow the system to sleep. recommend powertoy awake module
@echo off
REM ================================================
REM Batch script to run thermal_stitch_only.bat for multiple sites
REM Prevent system sleep: use PowerToys Awake or keep system active manually
REM ================================================

setlocal enabledelayedexpansion

REM ==== CLI args: %1 = BASE, %2.. = sites ====
if not "%~1"=="" set "BASE=%~1"
shift
if not "%~1"=="" set "OUTF=%~1"


set "SITES="
:__collect_sites
shift
if "%~1"=="" goto __sites_done
set "SITES=%SITES% %~1"
goto __collect_sites
:__sites_done

for %%S in (%SITES%) do (
    echo =====================================================
    echo [%%S] Patching orthomosaics ...
    echo =====================================================

    set "INPUT=%BASE%%%S\%OUTF%\"

    if not exist "!INPUT!" (
        echo [ERROR] Input folder not found: !INPUT!
        echo Skipping %%S...
        echo
    )

    REM run metashape for visual stitching
    call patch_orthomosaic_only.bat !INPUT!

    echo [INFO] Waiting 10 seconds before next job...
    powershell.exe -Command "Start-Sleep -Seconds 10"

    ::free memory before the run
    powershell -command "[System.GC]::Collect();"

)

echo =====================================================
echo All sites processed. You may close this window.
echo =====================================================

endlocal
pause
