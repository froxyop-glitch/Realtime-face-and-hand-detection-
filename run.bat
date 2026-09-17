@echo off
title "Real-Time Face, Age and Hand MoCap Tracker"
cd /d "%~dp0"
echo ===================================================
echo Starting Real-Time Motion Capture Tracker...
echo (Focus the video window and press 'q' to exit)
echo ===================================================
if exist ".venv\Scripts\python.exe" (
    call .venv\Scripts\python.exe tracker.py
) else (
    python tracker.py
)
pause
