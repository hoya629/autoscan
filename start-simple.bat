@echo off
title 문서 자동 입력 시스템 (간편 실행)
color 0A
echo.
echo ========================================
echo     문서 자동 입력 시스템 시작 중...
echo     (간편 실행 모드)
echo ========================================
echo.

:: npm dependencies 확인 후 설치
if not exist node_modules (
    echo 의존성 패키지를 설치합니다...
    call npm install
)

echo 프록시 서버와 프론트엔드를 동시에 시작합니다...
echo 브라우저가 자동으로 열립니다.
echo.
echo [주의] Ctrl+C를 누르거나 창을 닫으면 앱이 종료됩니다.
echo.

npm run start