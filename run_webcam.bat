@echo off
title Indian Sign Language (ISL) Webcam HUD
echo =====================================================================
echo Launching ISL Real-Time Webcam HUD...
echo =====================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "src\webcam_app.py"
) else (
    echo [ERROR] Virtual environment .venv not found.
    pause
)

if errorlevel 1 (
    echo.
    echo [ERROR] Webcam app closed with an error.
    pause
)
