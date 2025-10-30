
REM NOTE - thermal_preprocess.bat makes unnecessary MEDIA.txt in the folder. it does not harm anything.
REM         but you have to look at github/blackshale/drone-image-process/preproc_folder.sh





::free memory before the run
powershell -command "[System.GC]::Collect();"

echo %1 = site folder


:: --------------------------------
:: PREPROCESS IMGAES
:: --------------------------------
::  path conversion (dos to linux)
:: input 
set "inppath=%~1"
set "inppath=%inppath:\=/%"
set "inppath=%inppath:D:/=/mnt/d/%"

:: Preprocess: organize folders (classify thermal and optical photos)
call :preproc %inppath%

:: Preprocess: conver r-jpg to tif (T Celcius)
call :conv "%inppath%T"


goto :eof


::--------------------
:: FUNCTIONS
::--------------------

:preproc
wsl bash ~/github/drone-image-process/preproc_folder.sh "%~1"
goto :eof


:conv
wsl bash ~/github/drone-image-process/conv_rjpg2tif.sh "%~1"
goto :eof
