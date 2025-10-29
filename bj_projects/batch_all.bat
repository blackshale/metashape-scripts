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

REM ---- Site list from command line (starting at %2) ----
set "BASE=%~1"

REM Collect all arguments after %1 into SITES
set "SITES="
:collect_sites
shift
if "%~1"=="" goto sites_done
set "SITES=%SITES% %~1"
goto collect_sites
:sites_done

if not defined SITES (
    echo [INFO] No sites provided. Using default list: SITE01 SITE02
    set "SITES=SITE01 SITE02"
)

for %%S in (%SITES%) do (
    echo =====================================================
    echo [%%S] Starting metashape stitching...
    echo =====================================================

    set "INPUT=%BASE%\%%S\"

    if not exist "!INPUT!" (
        echo [ERROR] Input folder not found: !INPUT!
        echo Skipping %%S...
        echo.
        goto :continue
    )

    REM preprocess (organize folders to W(visual),T(thermal). run DJI Thermal toolkit (T-C)
    call thermal_preprocess.bat "!INPUT!"
    echo [INFO] Waiting 10 seconds before next job...
    powershell.exe -Command "Start-Sleep -Seconds 10"

    REM preprocess for visual photo downscaling, resolution down 8000x6000 --> 1600x1200
    call visual_preprocess.bat "!INPUT!"
    echo [INFO] Waiting 10 seconds before next job...
    powershell.exe -Command "Start-Sleep -Seconds 10"

    REM run metashape for thermal stitching
    call thermal_stitch_only.bat "!INPUT!"
    echo [INFO] Waiting 10 seconds before next job...
    powershell.exe -Command "Start-Sleep -Seconds 10"

    REM run metashape for visual stitching
    call visual_stitch_only.bat "!INPUT!"
    echo [INFO] Waiting 10 seconds before next job...
    powershell.exe -Command "Start-Sleep -Seconds 10"

    :continue
)

echo =====================================================
echo All sites processed. You may close this window.
echo =====================================================

endlocal
pause
