@echo off
title "Push to GitHub (Realtime-face-and-hand-detection-)"
cd /d "%~dp0"
echo ===================================================
echo Pushing Realtime Face and Hand Detection to:
echo git@github.com:froxyop-glitch/Realtime-face-and-hand-detection-.git
echo ===================================================
git push -u origin main
echo ===================================================
pause
