@echo off
echo Installing proxy server dependencies...
cd /d "%~dp0"

REM Check if package-proxy.json exists
if not exist package-proxy.json (
    echo package-proxy.json not found!
    pause
    exit /b 1
)

REM Install dependencies if node_modules doesn't exist
if not exist node_modules_proxy (
    echo Installing dependencies...
    npm install --prefix . --package-lock false --no-package-lock --no-shrinkwrap express cors axios dotenv nodemon
    mkdir node_modules_proxy
    echo Dependencies installed > node_modules_proxy\installed.txt
)

echo Starting AI API proxy server...
node proxy-server.cjs

pause