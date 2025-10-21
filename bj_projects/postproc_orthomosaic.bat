echo "%1 = target_tif_filename"

::  path conversion (dos to linux)
:: input 
set "inppath=%~1"
set "inppath=%inppath:\=/%"
set "inppath=%inppath:D:/=/mnt/d/%"

:: Convert to plain temperature figure by matlab program

call :postproc "%inppath%" "%2" "%3" "%4"

goto :eof



::--------------------
:: FUNCTIONS
::--------------------

:postproc
wsl bash ~/github/drone-image-process/postproc_orthomosaic.sh "%~1" "%~2" "%~3" "%~4" 
goto :eof