@echo off
chcp 65001 >nul
title Auto Document Input System
color 0A
echo.
echo ========================================
echo  Starting Auto Document Input System...
echo ========================================
echo.

:: Store the current directory
set "CURRENT_DIR=%CD%"

:: Check for Node.js installation
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js is not installed.
    echo Please download and install Node.js from https://nodejs.org
    pause
    exit /b 1
)

:: Check for npm dependencies
if not exist "node_modules" (
    echo [INFO] Installing dependencies...
    call npm install
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install packages.
        pause
        exit /b 1
    )
)

:: Check for .env file and show guidance
if not exist ".env" (
    echo [INFO] API key configuration required.
    echo Please create a .env file using .env.example as template,
    echo or configure API keys through the app settings.
    echo.
)

echo [1/3] Starting proxy server...
start "Proxy Server" cmd /k "chcp 65001 >nul && echo Proxy server is running... (Closing this window will terminate the app) && node proxy-server.cjs"

:: Wait a moment for the proxy server to start
timeout /t 3 /nobreak >nul

echo [2/3] Starting front-end server...
echo [INFO] The browser will open automatically...
echo [INFO] Use the settings button to configure API keys if needed.
echo.
echo ========================================
echo  App running...
echo  Browser: http://localhost:3000
echo  Proxy:   http://localhost:3003
echo ========================================
echo.
echo [FEATURES]
echo - Multi-file PDF upload support
echo - Individual page removal
echo - Automatic update checking
echo - Secure API key management
echo.
echo [CAUTION] Closing this window or the proxy server window will terminate the app.
echo           To exit the app, press Ctrl+C or close the window.
echo.

:: Run the front-end server (in this window)
call npm run dev

echo.
echo App has been terminated.
pause