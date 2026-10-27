@echo off
setlocal
title Indian Sign Language (ISL) Recognition

echo ============================================================
echo   Indian Sign Language (ISL) Recognition System
echo ============================================================
echo.

REM 1. Check Python installation
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10 from https://www.python.org/downloads/
    echo (Make sure to check "Add Python to PATH" during installation)
    echo.
    pause
    exit /b 1
)

REM 2. Create virtual environment and install dependencies if missing
if not exist ".venv\Scripts\python.exe" (
    echo [*] Initial setup: Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv.
        pause
        exit /b 1
    )
    echo [*] Installing dependencies from requirements.txt...
    .venv\Scripts\pip.exe install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install required packages.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed successfully!
    echo.
)

REM 3. Launch Desktop UI App
echo [*] Starting ISL Recognition Desktop App...
.venv\Scripts\python.exe src\ui_app.py
