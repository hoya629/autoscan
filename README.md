# Auto Document Input System

A powerful AI-driven document processing system that extracts structured data from import settlement documents using multiple AI providers.

## ✨ Latest Features

### 🚀 Multi-file Upload Support
- Process multiple PDF files simultaneously
- Organized display by filename with page counts
- Individual page selection and removal

### 🔄 Automatic Updates
- GitHub-based update checking
- Notification system for new releases
- One-click update navigation

### 🔒 Enhanced API Key Security
- **AES-GCM 암호화**: Web Crypto API를 사용한 강력한 암호화
- **로컬 파일 저장**: 브라우저를 닫아도 키가 안전하게 보존
- **자동 로딩**: 앱 시작 시 암호화된 키 자동 복구

### 🤖 Multi-AI Support
- **Google Gemini** (2.5 Flash, Pro models)
- **OpenAI** (o4-mini, GPT-4.1)
- **Claude** (Sonnet 4, Opus 4 via proxy)
- **Upstage** (Document Parse, Solar DocVision)
- **Local Models** (Ollama, LM Studio, LocalAI)

### 📊 Usage Analytics
- Track model performance and costs
- Export usage logs to CSV
- Model preference scoring

## 🚀 Quick Start

### Prerequisites
- Node.js (v16 or higher)
- Modern web browser

### Installation & Running

#### Windows
```bash
# Double-click the batch file
start-app.bat
```

#### Mac/Linux
```bash
# Make script executable and run
chmod +x start-app.sh
./start-app.sh
```

### API Key Configuration

Configure API keys using one of these methods:

1. **Through App Interface** (Recommended):
   - Click the "API 설정" button in the app
   - Enter your API keys securely
   - Keys are **AES-GCM encrypted** and stored locally
   - **Persistent storage**: Keys remain after browser restart

2. **Using .env file**:
   ```bash
   # Copy the example file
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

### Getting API Keys

- **Google Gemini**: [AI Studio](https://aistudio.google.com/app/apikey)
- **OpenAI**: [Platform API Keys](https://platform.openai.com/api-keys)
- **Claude**: [Anthropic Console](https://console.anthropic.com/)
- **Upstage**: [Console](https://console.upstage.ai/)

## 📖 Usage

1. **Select AI Provider**: Choose from available providers (only those with configured keys will be active)
2. **Upload Documents**: Drag & drop or click to upload multiple PDF files or images
3. **Select Pages**: Choose which pages to process from the thumbnails
4. **Remove Pages**: Hover over pages and click the × button to remove unwanted pages
5. **Process**: Click "데이터 추출" to extract structured data
6. **Edit & Export**: Modify data as needed and copy to Excel

## 🏠 Local AI Setup

### Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a vision model
ollama pull llama3.2-vision:11b
ollama pull llava:13b
```

### LM Studio
1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Load a vision-capable model
3. Start local server on port 1234

## 🛠️ Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Start proxy server (separate terminal)
npm run proxy

# Build for production
npm run build
```

## 🏗️ Architecture

- **Frontend**: Vanilla TypeScript with Vite
- **PDF Processing**: pdf.js library
- **AI Integration**: Direct API calls with proxy server for CORS
- **Security**: Client-side encryption for API keys
- **Updates**: GitHub Releases API integration

## 📁 Supported File Formats

- **Images**: JPG, PNG, GIF, WebP
- **Documents**: PDF (multi-page with individual page selection)

## 🔒 Security Features

### 강화된 API 키 보안
- **AES-GCM 256-bit 암호화**: Web Crypto API 사용
- **로컬 저장소**: localStorage에 암호화되어 저장
- **자동 복구**: 브라우저 재시작 시 자동 로드
- **키 우선순위**: UI 입력 키 → 환경변수 키

### 일반 보안
- 파일은 브라우저에서만 처리 (서버 전송 없음)
- AI API 외 외부 서버로 민감 데이터 전송 없음
- .env 파일을 커밋하지 마세요

## 🔄 Update System

The app automatically checks for updates from GitHub releases:
- Checks once per day
- Shows notification for new versions
- Click "업데이트" to visit the release page
- Manual update by downloading the latest release

## 📊 System Requirements

- Node.js 18+
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection (for AI API calls)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is proprietary software. All rights reserved.

## 🆘 Support

For issues and support, please create an issue in the GitHub repository.