# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Korean document processing application that uses multiple AI providers (Google Gemini, OpenAI, Claude, Upstage) to extract data from import settlement documents (수입 정산서). The app processes uploaded images or PDF files and automatically extracts structured data into a table format that can be copied to Excel.

## Key Features

- **Multi-AI Support**: Gemini, OpenAI, Claude (with proxy server), Upstage, and Local Models
- **Local AI Support**: Ollama, LM Studio, LocalAI integration
- **Fully Local Operation**: File processing occurs entirely in the browser, API calls only on demand
- **Easy Desktop Execution**: Simple batch file execution for end users
- **Real-time Editing**: Extracted data can be edited immediately
- **Excel Integration**: Clipboard copy for direct Excel pasting

## Development Commands

```bash
# Install dependencies
npm install

# Run development server (opens browser automatically)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Start proxy server for Claude/OpenAI APIs (separate terminal)
node proxy-server.cjs
# OR use the helper scripts:
# Windows: start-proxy.bat
# Mac/Linux: ./start-proxy.sh
```

## Quick Start for End Users

1. Double-click `start-app.bat` (Windows) or run `./start-app.sh` (Mac/Linux)
2. The application will automatically:
   - Check for Node.js installation
   - Install dependencies if needed
   - Start both proxy server and front-end server
   - Open browser at http://localhost:3000
3. Configure API keys using one of these methods:
   - Create a `.env` file using `.env.example` as template, OR
   - Click the "API Settings" button in the app interface
4. Select your preferred AI provider from the dropdown
5. Upload documents (now supports multiple PDF files at once)
6. Select pages to process and click "데이터 추출"

## Local AI Setup

### Ollama Setup
1. Install Ollama from https://ollama.ai
2. Pull a vision model: `ollama pull llama3.2-vision:11b` or `ollama pull llava:13b`
3. Ollama runs on http://localhost:11434 by default
4. The application will automatically detect available models

### LM Studio Setup
1. Install LM Studio from https://lmstudio.ai
2. Download a vision-capable model (e.g., LLaVA models)
3. Start the local server on port 1234
4. Configure endpoint if using different port

### LocalAI Setup
1. Install LocalAI from https://localai.io
2. Configure with a vision model
3. Start server on http://localhost:8080
4. Update .env file if using different endpoint

## Architecture

### Core Components

- **Multi-AI Processing**: Supports Google Gemini, OpenAI GPT-4, Claude APIs, and Upstage Document AI
- **Local File Handling**: Complete browser-based file processing with no server dependency
- **Settings Management**: LocalStorage-based configuration persistence
- **PDF Processing**: pdf.js library for multi-page PDF handling with selective processing
- **Data Management**: In-memory table state with real-time editing capabilities

### Key Files

- `index.tsx`: Main application logic with multi-AI support
- `index.html`: Single-page application with Korean UI and AI settings
- `index.css`: Responsive styling with settings UI
- `start.bat`: Windows batch file for easy execution
- `vite.config.ts`: Development server configuration
- `tsconfig.json`: TypeScript configuration

### Data Flow

1. API keys configured in .env file → Application reads environment variables and shows provider availability
2. User selects AI provider → Application validates API key availability and updates UI
3. User uploads image/PDF → Local file processing (preview/page selection)
4. User clicks "데이터 추출" → Selected AI API processes files
5. Extracted data added to table → Real-time editing enabled
6. Data formatted for Excel → Clipboard copy functionality

### AI Integration

- **Google Gemini**: Direct API integration with structured JSON schema (2.5 Flash/Pro models)
- **OpenAI**: Direct browser API with dangerouslyAllowBrowser flag (GPT-4o/4.1 models)
- **Claude**: Proxy server integration to handle CORS restrictions (Sonnet 4/Opus 4 models)
- **Upstage**: Direct API integration with model-specific endpoints (Solar models)
- **Ollama**: Local model integration with automatic model discovery (Llama 3.2 Vision, LLaVA, Moondream)
- **LM Studio**: Local model server integration with custom vision models
- **LocalAI**: Local OpenAI-compatible API server integration

### API Schema

All AI providers extract the same structured data:
- `date`: Document date (YYYY-MM-DD format)
- `quantity`: Amount in GT units
- `amountUSD`: Commercial invoice charge in USD
- `commissionUSD`: Commission amount in USD
- `totalUSD`: Total amount in USD
- `totalKRW`: Total amount in KRW
- `balanceKRW`: Balance amount in KRW

## Technical Stack

- **Frontend**: Vanilla TypeScript with DOM manipulation
- **Build Tool**: Vite with auto-browser opening
- **AI Integration**: Google Gemini API + OpenAI API
- **PDF Processing**: pdf.js (pdfjs-dist)
- **Module System**: ES modules with CDN imports
- **Storage**: Browser localStorage for AI provider selection, with real-time API key validation
- **Execution**: Windows batch file for easy startup

## Configuration

- **Vite Config**: Auto-opens browser on port 3000
- **Environment Variables**: API keys and endpoints configured in .env file (VITE_GEMINI_API_KEY, VITE_OPENAI_API_KEY, VITE_CLAUDE_API_KEY, VITE_UPSTASH_API_KEY, VITE_OLLAMA_ENDPOINT, VITE_LMSTUDIO_ENDPOINT, VITE_LOCALAI_ENDPOINT)
- **Local Storage**: Persistent AI provider selection with availability status indicators
- **Local Model Discovery**: Automatic detection and listing of available models from local AI servers