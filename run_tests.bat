@echo off
title Brain Tumor — Test Runner
color 0A

echo ============================================================
echo   Brain Tumor Detection — Full Test Suite
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/2] Model Unit Tests chal rahe hain...
echo ----------------------------------------
python test_model.py
set MODEL_EXIT=%ERRORLEVEL%

echo.
echo [2/2] Flask API Tests chal rahe hain...
echo ----------------------------------------
python test_api.py
set API_EXIT=%ERRORLEVEL%

echo.
echo ============================================================
if %MODEL_EXIT%==0 (
    echo   Model Tests : PASS
) else (
    echo   Model Tests : FAIL  ^<-- dekho upar
)

if %API_EXIT%==0 (
    echo   API Tests   : PASS
) else (
    echo   API Tests   : FAIL  ^<-- dekho upar
)
echo ============================================================

if %MODEL_EXIT%==0 if %API_EXIT%==0 (
    echo.
    echo   Mubarak! Saare tests pass ho gaye.
) else (
    echo.
    echo   Kuch tests fail hue. Upar output dekho.
)

echo.
pause
