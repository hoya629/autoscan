#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Title output
echo -e "${BLUE}========================================"
echo "    Starting Auto Document Input System..."
echo -e "========================================${NC}"
echo

# Check Node.js installation
if ! command -v node &> /dev/null; then
    echo -e "${RED}[ERROR] Node.js is not installed.${NC}"
    echo "Please download and install Node.js from https://nodejs.org"
    exit 1
fi

# Check npm dependencies
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}[INFO] Installing dependencies...${NC}"
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to install packages.${NC}"
        exit 1
    fi
fi

# Check for .env file and show guidance
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}[INFO] API key configuration required.${NC}"
    echo "Please create a .env file using .env.example as template,"
    echo "or configure API keys through the app settings."
    echo
fi

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down app...${NC}"
    # Kill child processes
    if [ ! -z "$PROXY_PID" ]; then
        kill $PROXY_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    # Force kill processes using the ports
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    lsof -ti:3003 | xargs kill -9 2>/dev/null
    echo -e "${GREEN}App has been terminated.${NC}"
    exit 0
}

# Set signal traps
trap cleanup SIGINT SIGTERM EXIT

echo -e "${GREEN}[1/3] Starting proxy server...${NC}"
node proxy-server.cjs &
PROXY_PID=$!

# Wait for proxy server to start
sleep 3

echo -e "${GREEN}[2/3] Starting front-end server...${NC}"
echo -e "${BLUE}[INFO] The browser will open automatically...${NC}"
echo -e "${BLUE}[INFO] Use the settings button to configure API keys if needed.${NC}"
echo
echo -e "${BLUE}========================================"
echo " App running..."
echo " Browser: http://localhost:3000"
echo " Proxy:   http://localhost:3003"
echo -e "========================================${NC}"
echo
echo -e "${GREEN}[FEATURES]${NC}"
echo "- Multi-file PDF upload support"
echo "- Individual page removal"
echo "- Automatic update checking"
echo "- Secure API key management"
echo
echo -e "${YELLOW}[CAUTION] Press Ctrl+C to terminate the app.${NC}"
echo

# Run front-end server
npm run dev &
FRONTEND_PID=$!

# Wait for both servers to run
wait