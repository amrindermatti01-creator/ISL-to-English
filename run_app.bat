@echo off
title Indian Sign Language (ISL) Recognition & Translation Bridge
echo =====================================================================
echo Launching ISL Recognition and Translation Desktop Application...
echo =====================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "src\ui_app.py"
) else (
    echo [ERROR] Virtual environment .venv not found.
    echo Please make sure .venv exists in this directory.
    pause
)

if errorlevel 1 (
    echo.
    echo [ERROR] Application closed with an error.
    pause
)
