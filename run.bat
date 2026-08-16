@echo off
REM Launch skeleton_opencanvas (activate conda env and run main.py)
REM Double-click this file to start the app.

call C:\Users\sleep\anaconda3\Scripts\activate.bat env_skeleton_opencanvas
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate conda env: env_skeleton_opencanvas
    echo Check that Anaconda is installed at C:\Users\sleep\anaconda3
    echo and the env "env_skeleton_opencanvas" exists.
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
