@echo off
title AI Audio Forensics Laboratory - Web Interface
cd /d "%~dp0"
echo ========================================================
echo    AI Audio Forensics - Voice Authenticity System
echo    Phase 4 Interactive Forensic Dashboard
echo ========================================================
echo.
echo Activating Python 3.14.3 virtual environment (.venv314)...
call .venv314\Scripts\activate.bat
echo Launching Streamlit web interface...
python -m streamlit run app.py
pause

