#!/bin/bash

echo "Installing proxy server dependencies..."
cd "$(dirname "$0")"

# Check if package-proxy.json exists
if [ ! -f "package-proxy.json" ]; then
    echo "package-proxy.json not found!"
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules_proxy" ]; then
    echo "Installing dependencies..."
    npm install --no-package-lock --no-shrinkwrap express cors axios dotenv nodemon
    mkdir node_modules_proxy
    echo "Dependencies installed" > node_modules_proxy/installed.txt
fi

echo "Starting AI API proxy server..."
node proxy-server.cjs