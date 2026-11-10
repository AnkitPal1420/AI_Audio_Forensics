@echo off
title AI Audio Forensics Laboratory - Test Suite
cd /d "%~dp0"
echo ========================================================
echo    AI Audio Forensics - Automated Verification Suite
echo ========================================================
echo.
echo Activating Python 3.14.3 virtual environment (.venv314)...
call .venv314\Scripts\activate.bat
echo Running complete pytest verification suite...
python -m pytest -v
echo.
echo Verifying codebase syntax and compilation...
python -m compileall app.py database src tests
echo.
pause

