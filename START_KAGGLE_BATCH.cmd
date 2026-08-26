@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_kaggle_batch.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Kaggle batch setup stopped with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
