@echo off
setlocal
title Local RAG Assistant

set "APP_DIR=%~dp0project"
set "PYTHON_EXE=%APP_DIR%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%~dp0..\project\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=py"
)

if not exist "%PYTHON_EXE%" if not "%PYTHON_EXE%"=="py" (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if "%PYTHON_EXE%"=="%APP_DIR%\.venv\Scripts\python.exe" goto run
if "%PYTHON_EXE%"=="%~dp0..\project\.venv\Scripts\python.exe" goto run
if "%PYTHON_EXE%"=="py" goto run
if "%PYTHON_EXE%"=="python" goto run

echo Python could not be found.
echo Install Python and the packages in project\requirements.txt first.
pause
exit /b 1

:run
cd /d "%APP_DIR%"
"%PYTHON_EXE%" web_app.py

if errorlevel 1 (
    echo.
    echo The local site stopped with an error.
    echo Check that Foundry Local and the required models are installed.
    pause
)
