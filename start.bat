@echo off
title 문서 자동 입력 시스템
color 0A
echo.
echo ========================================
echo     문서 자동 입력 시스템 시작 중...
echo ========================================
echo.

:: 현재 디렉토리를 저장
set CURRENT_DIR=%CD%

:: Node.js 설치 확인
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [오류] Node.js가 설치되지 않았습니다.
    echo Node.js를 https://nodejs.org 에서 다운로드하여 설치하세요.
    pause
    exit /b 1
)

:: npm dependencies 확인
if not exist node_modules (
    echo [정보] 의존성 패키지를 설치합니다...
    call npm install
    if %ERRORLEVEL% neq 0 (
        echo [오류] 패키지 설치에 실패했습니다.
        pause
        exit /b 1
    )
)

echo [1/3] 프록시 서버를 시작합니다...
start "프록시 서버" cmd /k "echo 프록시 서버 실행 중... (이 창을 닫으면 앱이 종료됩니다) && node proxy-server.cjs"

:: 프록시 서버가 시작될 때까지 잠시 대기
timeout /t 3 /nobreak >nul

echo [2/3] 프론트엔드 서버를 시작합니다...
echo [정보] 브라우저가 자동으로 열립니다...
echo.
echo ========================================
echo  앱 사용 중... 
echo  브라우저: http://localhost:3000
echo  프록시: http://localhost:3003
echo ========================================
echo.
echo [주의] 이 창이나 프록시 서버 창을 닫으면 앱이 종료됩니다.
echo        앱을 종료하려면 Ctrl+C를 누르거나 창을 닫으세요.
echo.

:: 프론트엔드 서버 실행 (이 창에서)
call npm run dev

echo.
echo 앱이 종료되었습니다.
pause