@echo off
title "Real-Time Face, Age and Hand MoCap Tracker"
cd /d "%~dp0"
echo ===================================================
echo Starting Real-Time Motion Capture Tracker...
echo (Focus the video window and press 'q' to exit)
echo ===================================================
call .venv\Scripts\python.exe tracker.py
pause
