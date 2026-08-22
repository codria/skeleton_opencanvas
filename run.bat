@echo off
setlocal
REM Launch skeleton_opencanvas: locate conda, activate the env, run main.py.
REM Double-click to start. No per-PC path editing needed.

REM ---- Change this only if your conda env has a different name ----
set "ENV_NAME=env_skeleton_opencanvas"

REM ---- Find a conda "activate.bat" across common install locations ----
set "ACTIVATE="
call :find_activate "%CONDA_ROOT%"
call :find_activate "%USERPROFILE%\anaconda3"
call :find_activate "%USERPROFILE%\miniconda3"
call :find_activate "%USERPROFILE%\Anaconda3"
call :find_activate "%USERPROFILE%\Miniconda3"
call :find_activate "%LOCALAPPDATA%\anaconda3"
call :find_activate "%LOCALAPPDATA%\miniconda3"
call :find_activate "%PROGRAMDATA%\anaconda3"
call :find_activate "%PROGRAMDATA%\miniconda3"
call :find_activate "%SYSTEMDRIVE%\ProgramData\Anaconda3"
REM Derive from CONDA_EXE when conda is already initialized on PATH
if not defined ACTIVATE if defined CONDA_EXE (
    for %%I in ("%CONDA_EXE%\..\..") do call :find_activate "%%~fI"
)

if not defined ACTIVATE (
    echo.
    echo [ERROR] Could not find a conda installation.
    echo Tried common locations under your user folder and ProgramData.
    echo Fix: install Anaconda/Miniconda, or point CONDA_ROOT at your conda folder,
    echo      e.g.  set "CONDA_ROOT=D:\tools\miniconda3"  then run this again.
    echo.
    pause
    exit /b 1
)

call "%ACTIVATE%" %ENV_NAME%
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate conda env: %ENV_NAME%
    echo Found conda at: %ACTIVATE%
    echo Check that the env exists ^(run: conda env list^).
    echo.
    pause
    exit /b 1
)

cd /d "%~dp0"
python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] main.py exited with an error.
    pause
    exit /b 1
)

exit /b 0

:find_activate
REM %~1 = candidate conda root. Sets ACTIVATE if its Scripts\activate.bat exists.
if defined ACTIVATE goto :eof
if "%~1"=="" goto :eof
if exist "%~1\Scripts\activate.bat" set "ACTIVATE=%~1\Scripts\activate.bat"
goto :eof
