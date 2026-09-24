@echo off
title "CyberDrive - One-Hand MoCap & Gesture Arcade Racer"
cd /d "%~dp0"
echo ==============================================================================
echo             CYBERDRIVE ARCADE RACING GAME (ONE-HAND EDITION)
echo ==============================================================================
echo One-Hand Driving Controls:
echo   [STEERING]   : Tilt your single hand left/right (or move hand laterally)
echo   [GAS]        : Open Palm (accelerate)
echo   [BRAKE]      : Clench Fist (slow down)
echo   [NITRO]      : Thumbs Up / Peace Sign (or Spacebar)
echo   [DRIFT]      : Pinch Fingers
echo.
echo Side Options & Hotkeys:
echo   [C]          : Toggle Controls Sidebar on/off
echo   [M]          : Switch between 1-Hand Mode and 2-Hands Mode
echo   [R]          : Restart Game (or flash Open Palm when crashed)
echo   [Q]          : Quit CyberDrive
echo   [KEYBOARD]   : A/D or Left/Right to steer, W to gas, S to brake
echo ==============================================================================
echo Starting CyberDrive...
call .venv\Scripts\python.exe cargame.py
pause
