@echo off
title "CyberDrive - Real-Time MoCap & Gesture Arcade Racer"
cd /d "%~dp0"
echo ==============================================================================
echo                      CYBERDRIVE ARCADE RACING GAME
echo ==============================================================================
echo Controls:
echo   [STEERING]   : Hold both hands up and tilt left/right (or move 1 hand)
echo   [GAS]        : Open Palm
echo   [BRAKE]      : Clench Fist
echo   [NITRO]      : Thumbs Up / Peace Sign (or Spacebar)
echo   [DRIFT]      : Pinch Fingers
echo   [KEYBOARD]   : A/D or Left/Right to steer, W to accelerate, S to brake
echo   [RESTART]    : Press 'R'
echo   [EXIT]       : Press 'Q'
echo ==============================================================================
echo Starting CyberDrive...
call .venv\Scripts\python.exe cargame.py
pause
