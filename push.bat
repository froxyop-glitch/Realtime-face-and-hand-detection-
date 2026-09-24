@echo off
title "Push to GitHub (openCV-work)"
cd /d "%~dp0"
set "PATH=%PATH%;C:\Users\shoho\AppData\Local\Programs\MinGit\cmd"
echo ===================================================
echo Pushing Motion Tracker to:
echo https://github.com/froxyop-glitch/openCV-work
echo ===================================================
git push -u origin main
echo ===================================================
pause
