# GitHub Repository Setup Guide

이 가이드는 자동 업데이트 시스템을 활성화하기 위해 GitHub repository를 설정하는 방법을 설명합니다.

## 1. GitHub Repository 생성

1. **GitHub에 로그인** 후 새 repository 생성
2. **Repository 이름**: `auto-scan-app` (또는 원하는 이름)
3. **설정**:
   - ✅ Public (자동 업데이트를 위해 권장)
   - ✅ Add a README file 선택 해제 (이미 있음)
   - ✅ Add .gitignore 선택 해제 (이미 있음)

## 2. 로컬 Git 초기화 및 업로드

```bash
# 1. 프로젝트 폴더에서 Git 초기화
cd /path/to/your/project
git init

# 2. GitHub repository를 remote로 추가
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 3. .gitignore가 제대로 작동하는지 확인
git add .
git status

# 4. 첫 번째 커밋 생성
git commit -m "Initial commit: Auto Document Input System v1.0.0

- Multi-file PDF upload support
- Individual page removal functionality  
- Automatic GitHub update checking
- Secure API key management
- Multi-AI provider support (Gemini, OpenAI, Claude, Upstage, Local)"

# 5. main 브랜치로 변경 (GitHub 기본값)
git branch -M main

# 6. GitHub에 업로드
git push -u origin main
```

## 3. 코드에서 Repository 정보 업데이트

`index.tsx` 파일에서 다음 라인을 수정하세요:

```typescript
// 현재
githubRepo: 'your-username/auto-scan-app', // TODO: Replace with actual GitHub repository

// 수정 후
githubRepo: 'YOUR_GITHUB_USERNAME/YOUR_REPO_NAME',
```

예시:
```typescript
githubRepo: 'johndoe/auto-document-scanner',
```

## 4. 첫 번째 릴리스 생성

### GitHub 웹사이트에서:

1. **Repository 페이지**로 이동
2. **"Releases"** 탭 클릭
3. **"Create a new release"** 클릭
4. **릴리스 정보 입력**:
   ```
   Tag version: v1.0.0
   Release title: Auto Document Input System v1.0.0
   
   Description:
   🚀 Initial Release
   
   ## ✨ Features
   - Multi-file PDF upload and processing
   - Individual PDF page selection and removal
   - Multi-AI provider support (Gemini, OpenAI, Claude, Upstage)
   - Local AI integration (Ollama, LM Studio, LocalAI)
   - Secure API key management with encryption
   - Real-time data editing and Excel export
   - Usage analytics and cost tracking
   - Automatic update checking
   
   ## 🛠️ Setup
   1. Download and extract the release
   2. Run `start-app.bat` (Windows) or `./start-app.sh` (Mac/Linux)
   3. Configure API keys through the settings panel
   4. Upload documents and start processing!
   
   ## 📋 Requirements
   - Node.js 16+
   - Modern web browser
   ```

5. **"Publish release"** 클릭

### Command Line에서 (GitHub CLI 사용):

```bash
# GitHub CLI 설치 후
gh release create v1.0.0 \
    --title "Auto Document Input System v1.0.0" \
    --notes "🚀 Initial release with multi-file upload, AI integration, and automatic updates"
```

## 5. 자동 업데이트 테스트

1. **앱 실행** 후 2초 정도 기다리기
2. **업데이트 알림**이 상단에 나타나는지 확인
3. **"업데이트"** 버튼 클릭 시 GitHub 릴리스 페이지로 이동하는지 확인

## 6. 새 버전 배포 워크플로

새 버전을 배포할 때:

1. **코드 업데이트**
   ```bash
   git add .
   git commit -m "feat: new features description"
   git push
   ```

2. **버전 업데이트**
   - `package.json`의 `version` 필드 업데이트
   - `index.tsx`의 `APP_CONFIG.version` 업데이트

3. **새 릴리스 생성**
   ```bash
   git tag v1.0.1
   git push --tags
   gh release create v1.0.1 --title "Auto Document Input System v1 --generate-notes
   ```

4. **사용자들은 자동으로 업데이트 알림을 받음**

## 7. 선택사항: GitHub Actions로 자동화

`.github/workflows/release.yml` 파일을 생성하여 자동 배포:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Create Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          generate_release_notes: true
```

## 보안 고려사항

- **API 키는 절대 커밋하지 마세요** (.env 파일은 .gitignore에 포함됨)
- **Private repository**를 사용하는 경우 업데이트 체크가 실패할 수 있습니다
- **사용자 데이터**는 로컬에만 저장되며 GitHub에 업로드되지 않습니다

## 문제 해결

### 업데이트 알림이 나타나지 않는 경우:
1. **Repository가 public**인지 확인
2. **Repository 이름**이 코드에 올바르게 설정되었는지 확인
3. **릴리스가 생성**되었는지 확인
4. **브라우저 개발자 도구**에서 네트워크 오류 확인

### 403 Forbidden 오류:
1. **API rate limit** 도달 (시간당 60회 제한)
2. **Repository가 private**일 수 있음
3. **Repository 이름**이 잘못되었을 수 있음