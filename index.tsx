import { GoogleGenAI, Type } from "@google/genai";
import OpenAI from "openai";
import * as pdfjsLib from "pdfjs-dist";

// Set worker source for pdf.js
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://esm.sh/pdfjs-dist@4.4.168/build/pdf.worker.mjs`;

// Data structure for a table row
interface TableRowData {
    id: number;
    date: string;
    quantity: number | string;
    amountUSD: number | string;
    commissionUSD: number | string;
    totalUSD: number | string;
    totalKRW: number | string;
    balanceKRW: number | string;
}

interface PageData {
    data: string;
    mimeType: string;
    fileName: string;
    pageNumber?: number; // For PDF pages
}

interface AISettings {
    provider: 'gemini' | 'openai' | 'upstage';
    model: string;
}

interface UsageLog {
    id: string;
    timestamp: Date;
    provider: string;
    model: string;
    processingTime: number; // in milliseconds
    pagesProcessed: number;
    inputTokens: number;
    outputTokens: number;
    totalCostUSD: number;
    rating?: 'like' | 'dislike';
    ratedAt?: Date;
}

interface ModelStats {
    provider: string;
    model: string;
    totalUsage: number;
    totalPages: number;
    averageProcessingTime: number;
    totalInputTokens: number;
    totalOutputTokens: number;
    totalCostUSD: number;
    averageCostPerPage: number;
    likeCount: number;
    dislikeCount: number;
    preferenceScore: number; // (likes - dislikes) / totalUsage
}

interface DeleteAction {
    pageData: PageData;
    timestamp: number;
}

// Delete history for undo functionality
const deleteHistory: DeleteAction[] = [];
const MAX_UNDO_HISTORY = 10;



// Pricing information (USD per 1M tokens) - Updated for 2025
const MODEL_PRICING = {
    // Gemini pricing
    'gemini-2.5-flash': { input: 0.075, output: 0.30 },
    'gemini-2.5-pro': { input: 1.25, output: 5.00 },
    'gemini-2.5-flash-lite-preview-06-17': { input: 0.075, output: 0.30 },
    
    // OpenAI pricing
    'o4-mini': { input: 0.15, output: 0.60 }, // Estimated pricing for o4-mini
    'gpt-4.1': { input: 10.00, output: 30.00 }, // Estimated pricing for GPT-4.1
    'gpt-4o-mini': { input: 0.15, output: 0.60 },
    'gpt-4o': { input: 2.50, output: 10.00 },
    
    
    // Upstage pricing (estimated)
    'document-parse': { input: 0.50, output: 1.00 },
} as const;

function calculateCost(modelId: string, inputTokens: number, outputTokens: number): number {
    const pricing = MODEL_PRICING[modelId as keyof typeof MODEL_PRICING];
    if (!pricing) return 0;
    
    const inputCost = (inputTokens / 1_000_000) * pricing.input;
    const outputCost = (outputTokens / 1_000_000) * pricing.output;
    return inputCost + outputCost;
}

function estimateTokens(text: string, isImage = false): number {
    if (isImage) {
        // Vision models typically use more tokens for images
        // Rough estimate: ~1000 tokens per image + text tokens
        return 1000 + Math.ceil(text.length / 4);
    }
    // Rough estimate: ~1 token per 4 characters
    return Math.ceil(text.length / 4);
}

// Available models for each provider (updated for 2025)
const PROVIDER_MODELS = {
    gemini: [
        { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', description: '최고 가성비 모델' },
        { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro', description: '최고 성능 사고 모델' },
        { id: 'gemini-2.5-flash-lite-preview-06-17', name: 'Gemini 2.5 Flash Lite', description: '최저 비용 고속 모델' }
    ],
    openai: [
        { id: 'o4-mini', name: 'o4-mini', description: '차세대 소형 모델' },
        { id: 'gpt-4.1', name: 'GPT-4.1', description: '최신 GPT-4.1 모델' }
    ],
    upstage: [
        { id: 'document-parse', name: 'Document Parse', description: '문서 텍스트 추출 모델' }
    ]
};

// Application configuration
const APP_CONFIG = {
    version: '1.0.1',
    githubRepo: '', // Will be auto-detected
    checkForUpdates: true
};

// Global state
let tableData: TableRowData[] = [];
let nextId = 0;
let selectedPages: PageData[] = [];
let pdfFileGroups: Map<string, PageData[]> = new Map(); // Group pages by filename
let currentSettings: AISettings = {
    provider: 'gemini',
    model: 'gemini-2.5-flash'
};

// Usage logging
let usageLogs: UsageLog[] = [];
let currentLogId: string | null = null;


// Load settings from localStorage
function loadSettings(): AISettings {
    const stored = localStorage.getItem('aiSettings');
    if (stored) {
        try {
            const parsed = JSON.parse(stored);
            const provider = parsed.provider || 'gemini';
            const model = parsed.model || getDefaultModelForProvider(provider);
            return { provider, model };
        } catch (e) {
            console.error('Error loading settings:', e);
        }
    }
    return { provider: 'gemini', model: 'gemini-2.5-flash' };
}

// Get default model for a provider
function getDefaultModelForProvider(provider: string): string {
    const models = PROVIDER_MODELS[provider as keyof typeof PROVIDER_MODELS];
    return models?.[0]?.id || 'gemini-2.5-flash';
}

// Get available models for current provider
function getAvailableModels(): Array<{id: string, name: string, description: string}> {
    if (!isProviderAvailable(currentSettings.provider)) {
        return [];
    }
    const models = PROVIDER_MODELS[currentSettings.provider as keyof typeof PROVIDER_MODELS];
    return models ? [...models] : [];
}

// Save settings to localStorage
function saveSettings(settings: AISettings) {
    localStorage.setItem('aiSettings', JSON.stringify(settings));
    currentSettings = settings;
}

// Usage logging functions
function loadUsageLogs(): UsageLog[] {
    const stored = localStorage.getItem('usageLogs');
    if (stored) {
        try {
            const parsed = JSON.parse(stored);
            return parsed.map((log: any) => ({
                ...log,
                timestamp: new Date(log.timestamp),
                ratedAt: log.ratedAt ? new Date(log.ratedAt) : undefined,
                // Add default values for new fields (backward compatibility)
                inputTokens: log.inputTokens || 0,
                outputTokens: log.outputTokens || 0,
                totalCostUSD: log.totalCostUSD || 0
            }));
        } catch (e) {
            console.error('Error loading usage logs:', e);
        }
    }
    return [];
}

function saveUsageLogs() {
    localStorage.setItem('usageLogs', JSON.stringify(usageLogs));
}

function startLogging(provider: string, model: string, pagesCount: number): string {
    const logId = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const log: UsageLog = {
        id: logId,
        timestamp: new Date(),
        provider,
        model,
        processingTime: 0,
        pagesProcessed: pagesCount,
        inputTokens: 0,
        outputTokens: 0,
        totalCostUSD: 0
    };
    usageLogs.push(log);
    currentLogId = logId;
    return logId;
}

function endLogging(logId: string, processingTime: number, inputTokens = 0, outputTokens = 0) {
    const log = usageLogs.find(l => l.id === logId);
    if (log) {
        log.processingTime = processingTime;
        log.inputTokens = inputTokens;
        log.outputTokens = outputTokens;
        log.totalCostUSD = calculateCost(log.model, inputTokens, outputTokens);
        saveUsageLogs();
    }
    currentLogId = null;
}

function addTokensToCurrentLog(inputTokens: number, outputTokens: number) {
    if (currentLogId) {
        const log = usageLogs.find(l => l.id === currentLogId);
        if (log) {
            log.inputTokens += inputTokens;
            log.outputTokens += outputTokens;
            log.totalCostUSD = calculateCost(log.model, log.inputTokens, log.outputTokens);
        }
    }
}

function rateLastResult(rating: 'like' | 'dislike') {
    if (usageLogs.length > 0) {
        const lastLog = usageLogs[usageLogs.length - 1];
        lastLog.rating = rating;
        lastLog.ratedAt = new Date();
        saveUsageLogs();
        updateRatingButtons();
    }
}

function calculateModelStats(): ModelStats[] {
    const statsMap = new Map<string, ModelStats>();
    
    usageLogs.forEach(log => {
        const key = `${log.provider}_${log.model}`;
        if (!statsMap.has(key)) {
            statsMap.set(key, {
                provider: log.provider,
                model: log.model,
                totalUsage: 0,
                totalPages: 0,
                averageProcessingTime: 0,
                totalInputTokens: 0,
                totalOutputTokens: 0,
                totalCostUSD: 0,
                averageCostPerPage: 0,
                likeCount: 0,
                dislikeCount: 0,
                preferenceScore: 0
            });
        }
        
        const stats = statsMap.get(key)!;
        stats.totalUsage++;
        stats.totalPages += log.pagesProcessed;
        stats.averageProcessingTime = ((stats.averageProcessingTime * (stats.totalUsage - 1)) + log.processingTime) / stats.totalUsage;
        stats.totalInputTokens += log.inputTokens;
        stats.totalOutputTokens += log.outputTokens;
        stats.totalCostUSD += log.totalCostUSD;
        stats.averageCostPerPage = stats.totalPages > 0 ? stats.totalCostUSD / stats.totalPages : 0;
        
        if (log.rating === 'like') stats.likeCount++;
        if (log.rating === 'dislike') stats.dislikeCount++;
        
        stats.preferenceScore = stats.totalUsage > 0 ? (stats.likeCount - stats.dislikeCount) / stats.totalUsage : 0;
    });
    
    return Array.from(statsMap.values()).sort((a, b) => b.preferenceScore - a.preferenceScore);
}

function updateRatingButtons() {
    const lastLog = usageLogs[usageLogs.length - 1];
    if (!lastLog) return;
    
    likeButton.classList.toggle('selected', lastLog.rating === 'like');
    dislikeButton.classList.toggle('selected', lastLog.rating === 'dislike');
}

function showRatingSection() {
    ratingSection.classList.remove('hidden');
    updateRatingButtons();
}

function hideRatingSection() {
    ratingSection.classList.add('hidden');
}

function renderAnalyticsDashboard() {
    const stats = calculateModelStats();
    
    // 최근 사용 기록 렌더링
    renderRecentUsage();
    
    if (stats.length === 0) {
        costSummary.innerHTML = '';
        modelStatsTable.innerHTML = '<p>아직 사용 기록이 없습니다.</p>';
        return;
    }
    
    // Render cost summary
    const totalCost = stats.reduce((sum, stat) => sum + stat.totalCostUSD, 0);
    const totalPages = stats.reduce((sum, stat) => sum + stat.totalPages, 0);
    const totalUsages = stats.reduce((sum, stat) => sum + stat.totalUsage, 0);
    
    costSummary.innerHTML = `
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <h4 style="margin: 0 0 0.5rem 0;">💰 전체 비용 요약</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                <div>
                    <strong>총 비용:</strong><br>
                    <span style="font-size: 1.2em; color: #dc3545;">$${totalCost.toFixed(4)}</span>
                </div>
                <div>
                    <strong>총 사용 횟수:</strong><br>
                    <span style="font-size: 1.2em; color: #28a745;">${totalUsages}회</span>
                </div>
                <div>
                    <strong>총 처리 페이지:</strong><br>
                    <span style="font-size: 1.2em; color: #007bff;">${totalPages}페이지</span>
                </div>
                <div>
                    <strong>평균 페이지당 비용:</strong><br>
                    <span style="font-size: 1.2em; color: #6f42c1;">$${totalPages > 0 ? (totalCost / totalPages).toFixed(4) : '0.0000'}</span>
                </div>
            </div>
        </div>
    `;
    
    const table = document.createElement('table');
    table.className = 'stats-table';
    
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>제공자</th>
            <th>모델</th>
            <th>사용 횟수</th>
            <th>총 페이지</th>
            <th>평균 처리시간</th>
            <th>총 비용 (USD)</th>
            <th>페이지당 비용</th>
            <th>총 토큰</th>
            <th>좋아요</th>
            <th>싫어요</th>
            <th>선호도 점수</th>
        </tr>
    `;
    table.appendChild(thead);
    
    const tbody = document.createElement('tbody');
    stats.forEach(stat => {
        const row = document.createElement('tr');
        const scoreClass = stat.preferenceScore > 0 ? 'positive' : stat.preferenceScore < 0 ? 'negative' : 'neutral';
        const totalTokens = stat.totalInputTokens + stat.totalOutputTokens;
        
        row.innerHTML = `
            <td>${stat.provider}</td>
            <td>${stat.model}</td>
            <td>${stat.totalUsage}</td>
            <td>${stat.totalPages}</td>
            <td>${(stat.averageProcessingTime / 1000).toFixed(2)}초</td>
            <td>$${stat.totalCostUSD.toFixed(4)}</td>
            <td>$${stat.averageCostPerPage.toFixed(4)}</td>
            <td>${totalTokens.toLocaleString()}</td>
            <td>${stat.likeCount}</td>
            <td>${stat.dislikeCount}</td>
            <td class="preference-score ${scoreClass}">${stat.preferenceScore.toFixed(2)}</td>
        `;
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    
    modelStatsTable.innerHTML = '';
    modelStatsTable.appendChild(table);
}

function toggleAnalyticsDashboard() {
    const isHidden = analyticsDashboard.classList.contains('hidden');
    if (isHidden) {
        renderAnalyticsDashboard();
        analyticsDashboard.classList.remove('hidden');
        showAnalyticsButton.querySelector('span')!.textContent = '사용 통계 숨기기';
    } else {
        analyticsDashboard.classList.add('hidden');
        showAnalyticsButton.querySelector('span')!.textContent = '사용 통계 보기';
    }
}

function renderRecentUsage() {
    if (!recentUsageList) return;
    
    if (usageLogs.length === 0) {
        recentUsageList.innerHTML = '<div class="empty-recent-usage">아직 사용 기록이 없습니다.</div>';
        return;
    }
    
    // 최근 10개 기록만 표시 (최신순)
    const recentLogs = [...usageLogs]
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
        .slice(0, 10);
    
    recentUsageList.innerHTML = recentLogs.map(log => {
        const timeAgo = getTimeAgo(log.timestamp);
        const ratingIcon = log.rating === 'like' ? '👍' : log.rating === 'dislike' ? '👎' : '-';
        const ratingClass = log.rating === 'like' ? 'positive' : log.rating === 'dislike' ? 'negative' : '';
        
        return `
            <div class="recent-usage-item">
                <div class="recent-usage-main">
                    <div class="recent-usage-provider">${log.provider.toUpperCase()}</div>
                    <div class="recent-usage-model">${log.model}</div>
                </div>
                <div class="recent-usage-details">
                    <span class="recent-usage-time">${timeAgo}</span>
                    <span class="recent-usage-cost">$${(log.totalCostUSD || 0).toFixed(4)}</span>
                    <span class="recent-usage-rating ${ratingClass}">${ratingIcon}</span>
                </div>
            </div>
        `;
    }).join('');
}

function getTimeAgo(timestamp: Date): string {
    const now = new Date();
    const diff = now.getTime() - new Date(timestamp).getTime();
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (minutes < 1) return '방금 전';
    if (minutes < 60) return `${minutes}분 전`;
    if (hours < 24) return `${hours}시간 전`;
    return `${days}일 전`;
}

function clearAllLogs() {
    if (usageLogs.length === 0) {
        alert('삭제할 로그가 없습니다.');
        return;
    }
    
    if (confirm('모든 사용 로그를 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.')) {
        usageLogs = [];
        localStorage.removeItem('usageLogs');
        renderAnalyticsDashboard();
        alert('모든 로그가 삭제되었습니다.');
    }
}

function exportUsageLogs() {
    if (usageLogs.length === 0) {
        alert('내보낼 로그가 없습니다.');
        return;
    }
    
    const csv = [
        ['타임스탬프', '제공자', '모델', '처리시간(초)', '페이지수', '입력토큰', '출력토큰', '비용(USD)', '평가', '평가일시'].join(','),
        ...usageLogs.map(log => [
            log.timestamp.toISOString(),
            log.provider,
            log.model,
            (log.processingTime / 1000).toFixed(2),
            log.pagesProcessed,
            log.inputTokens || 0,
            log.outputTokens || 0,
            (log.totalCostUSD || 0).toFixed(4),
            log.rating || '',
            log.ratedAt ? log.ratedAt.toISOString() : ''
        ].join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `usage_logs_${new Date().toISOString().slice(0, 10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// DOM Elements
const dropZone = document.getElementById('drop-zone') as HTMLDivElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const imagePreview = document.getElementById('image-preview') as HTMLImageElement;
const processButton = document.getElementById('process-button') as HTMLButtonElement;
const addRowButton = document.getElementById('add-row-button') as HTMLButtonElement;
const copyButton = document.getElementById('copy-button') as HTMLButtonElement;
const tableBody = document.querySelector('#data-table tbody') as HTMLTableSectionElement;
const loadingOverlay = document.getElementById('loading-overlay') as HTMLDivElement;
const pdfPreviewContainer = document.getElementById('pdf-preview-container') as HTMLDivElement;
const pdfPagesContainer = document.getElementById('pdf-pages') as HTMLDivElement;

// Rating and analytics elements
const ratingSection = document.getElementById('rating-section') as HTMLDivElement;
const likeButton = document.getElementById('like-button') as HTMLButtonElement;
const dislikeButton = document.getElementById('dislike-button') as HTMLButtonElement;
const showAnalyticsButton = document.getElementById('show-analytics-button') as HTMLButtonElement;
const analyticsDashboard = document.getElementById('analytics-dashboard') as HTMLDivElement;
const modelStatsTable = document.getElementById('model-stats-table') as HTMLDivElement;
const costSummary = document.getElementById('cost-summary') as HTMLDivElement;
const exportLogsButton = document.getElementById('export-logs-button') as HTMLButtonElement;
const clearLogsButton = document.getElementById('clear-logs-button') as HTMLButtonElement;
const recentUsageList = document.getElementById('recent-usage-list') as HTMLDivElement;

// Settings elements (will be initialized after DOM loads)
let aiProviderPills: NodeListOf<HTMLButtonElement>;
let modelSelector: HTMLSelectElement;
let tabButtons: NodeListOf<HTMLButtonElement>;

let debugStatusButton: HTMLButtonElement;

// Update elements
let updateNotification: HTMLDivElement;
let updateVersionSpan: HTMLSpanElement;
let updateButton: HTMLButtonElement;
let dismissUpdateButton: HTMLButtonElement;
let checkUpdateButton: HTMLButtonElement;

// API Settings Modal elements
let apiSettingsModal: HTMLDivElement;
let apiSettingsButton: HTMLButtonElement;
let closeModalButton: HTMLButtonElement;
let modalOverlay: HTMLDivElement;
let saveApiKeysButton: HTMLButtonElement;
let cancelApiSettingsButton: HTMLButtonElement;
let geminiKeyInput: HTMLInputElement;
let openaiKeyInput: HTMLInputElement;
let upstageKeyInput: HTMLInputElement;

// Check if provider is available (proxy server or local endpoint)
async function isProviderAvailable(provider: string): Promise<boolean> {
    // First check if API key is available for all providers except local ones
    if (['gemini', 'openai', 'upstage'].includes(provider)) {
        const apiKey = getAPIKey(provider);
        console.log(`[Provider Check] ${provider} API key available:`, !!apiKey, `(key: "${apiKey}")`);
        if (!apiKey || apiKey.trim() === '') {
            console.log(`[Provider Check] ${provider} API key is empty or null - returning false`);
            return false;
        }
        
        // For proxy server providers, check if proxy server is running (required for security)
        if (['gemini', 'openai', 'upstage'].includes(provider)) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 2000);
                
                const response = await fetch('http://localhost:3002/health', {
                    method: 'GET',
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                console.log(`✅ [Provider Check] Proxy server available for ${provider}: ${response.status}`);
                
                if (response.ok) {
                    console.log(`✅ [Provider Check] ${provider} ready to use with proxy server`);
                    return true;
                }
                
                console.log(`❌ [Provider Check] Proxy server responded with error for ${provider}: ${response.status}`);
                return false;
            } catch (error) {
                console.error(`❌ [Provider Check] Proxy server connection failed for ${provider}:`, error.message);
                console.error(`❌ [Provider Check] Please start the proxy server with: npm run proxy`);
                
                // For testing purposes, allow direct API calls if proxy is not available
                console.warn(`⚠️ [Provider Check] Allowing direct API calls for ${provider} (not recommended for production)`);
                return true; // Allow usage with direct API calls
            }
        }
        
        // Claude doesn't need proxy server, just API key
        return true;
    }
    
    
    return false;
}

// Synchronous version for backward compatibility
function isProviderAvailableSync(provider: string): boolean {
    // For proxy server providers, check if API key is available
    if (['gemini', 'openai', 'upstage'].includes(provider)) {
        const apiKey = getAPIKey(provider);
        return !!apiKey && apiKey.trim() !== '';
    }
    
    
    return false;
}

// Update provider pill status
async function updateProviderPillsStatus() {
    if (!aiProviderPills) return;
    
    console.log('🔄 [Provider Status] Starting provider status update...');
    
    // First check if proxy server is running
    try {
        const proxyResponse = await fetch('http://localhost:3002/health', { 
            method: 'GET',
            signal: AbortSignal.timeout(2000)
        });
        if (proxyResponse.ok) {
            console.log('✅ [Provider Status] Proxy server is running on port 3002');
        } else {
            console.error('❌ [Provider Status] Proxy server responded with error:', proxyResponse.status);
        }
    } catch (error) {
        console.error('❌ [Provider Status] Proxy server not accessible:', error.message);
        console.error('❌ [Provider Status] Please start proxy server with: npm run proxy');
    }
    
    for (const pill of aiProviderPills) {
        const provider = pill.dataset.provider as string;
        const statusIndicator = pill.querySelector('.pill-status') as HTMLElement;
        
        console.log(`🔍 [Provider Status] Checking ${provider}...`);
        
        // Debug API key availability
        const apiKey = getAPIKey(provider);
        console.log(`🔍 [Provider Status] ${provider} API key:`, apiKey ? `${apiKey.length} chars` : 'not found');
        
        const isAvailable = await isProviderAvailable(provider);
        console.log(`📊 [Provider Status] ${provider} available: ${isAvailable}`);
        
        // Remove existing classes
        pill.classList.remove('active', 'available', 'unavailable');
        statusIndicator.classList.remove('available', 'unavailable');
        
        // Update availability status
        if (isAvailable) {
            console.log(`✅ [Provider Status] Setting ${provider} as AVAILABLE`);
            pill.classList.add('available');
            statusIndicator.classList.add('available');
        } else {
            console.log(`❌ [Provider Status] Setting ${provider} as UNAVAILABLE`);
            pill.classList.add('unavailable');
            statusIndicator.classList.add('unavailable');
        }
        
        // Mark current selection
        if (provider === currentSettings.provider) {
            pill.classList.add('active');
        }
    }
    
    console.log('✅ [Provider Status] Provider status update completed');
}

// Update model selector based on current provider
function updateModelSelector() {
    if (!modelSelector) return;
    
    const availableModels = getAvailableModels();
    
    // Clear existing options
    modelSelector.innerHTML = '';
    
    if (availableModels.length === 0) {
        modelSelector.innerHTML = '<option value="">사용 가능한 제공자를 선택하세요</option>';
        modelSelector.disabled = true;
        return;
    }
    
    // Add model options
    availableModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = `${model.name} - ${model.description}`;
        
        if (model.id === currentSettings.model) {
            option.selected = true;
        }
        
        modelSelector.appendChild(option);
    });
    
    modelSelector.disabled = false;
    
    // If current model is not available for the provider, select the first one
    if (!availableModels.find(m => m.id === currentSettings.model)) {
        currentSettings.model = availableModels[0].id;
        modelSelector.value = currentSettings.model;
        saveSettings(currentSettings);
    }
}


// Initialize settings
function initializeSettings() {
    currentSettings = loadSettings();
    
    // Validate that the model exists for the provider
    const availableModels = PROVIDER_MODELS[currentSettings.provider as keyof typeof PROVIDER_MODELS];
    if (availableModels && !availableModels.find(m => m.id === currentSettings.model)) {
        currentSettings.model = getDefaultModelForProvider(currentSettings.provider);
        saveSettings(currentSettings);
    }
}

// Update checking functions

function compareVersions(version1: string, version2: string): number {
    const v1parts = version1.replace(/^v/, '').split('.').map(Number);
    const v2parts = version2.replace(/^v/, '').split('.').map(Number);
    
    for (let i = 0; i < Math.max(v1parts.length, v2parts.length); i++) {
        const v1part = v1parts[i] || 0;
        const v2part = v2parts[i] || 0;
        
        if (v1part < v2part) return -1;
        if (v1part > v2part) return 1;
    }
    return 0;
}


// API Key Security Functions

interface SecureAPIKeys {
    gemini?: string;
    openai?: string;
    upstage?: string;
}

// Cache for API keys
let cachedAPIKeys: SecureAPIKeys | null = null;

// GitHub Update System
interface GitHubRelease {
    tag_name: string;
    name: string;
    html_url: string;
    published_at: string;
    body: string;
}

// Auto-detect GitHub repository
async function detectGitHubRepository(): Promise<string> {
    try {
        // Try to read from package.json first
        const response = await fetch('/package.json');
        if (response.ok) {
            const packageJson = await response.json();
            if (packageJson.repository && packageJson.repository.url) {
                const url = packageJson.repository.url;
                const match = url.match(/github\.com[\/:](.+?)\.git/);
                if (match) {
                    return match[1];
                }
            }
        }
    } catch (error) {
        console.log('Could not read package.json:', error);
    }

    // Try common repository patterns based on current location
    const currentPath = window.location.pathname;
    const possibleRepos = [
        'hoya629/autoscan',
        'hoya629/auto-scan-updated',
        'hoya629/document-auto-input'
    ];

    // Test each repository to see which one exists
    for (const repo of possibleRepos) {
        try {
            const testResponse = await fetch(`https://api.github.com/repos/${repo}`);
            if (testResponse.ok) {
                console.log(`🔍 Auto-detected GitHub repository: ${repo}`);
                return repo;
            }
        } catch (error) {
            // Continue to next repo
        }
    }

    // Default fallback
    return 'hoya629/autoscan';
}

async function checkForUpdates(): Promise<GitHubRelease | null> {
    try {
        // Auto-detect repository if not set
        if (!APP_CONFIG.githubRepo) {
            APP_CONFIG.githubRepo = await detectGitHubRepository();
        }

        console.log(`Checking for updates from GitHub repository: ${APP_CONFIG.githubRepo}`);
        
        const response = await fetch(`https://api.github.com/repos/${APP_CONFIG.githubRepo}/releases/latest`);
        
        if (!response.ok) {
            console.log('GitHub API request failed:', response.status);
            return null;
        }
        
        const release: GitHubRelease = await response.json();
        const latestVersion = release.tag_name.replace(/^v/, ''); // Remove 'v' prefix if present
        const currentVersion = APP_CONFIG.version;
        
        console.log(`Current version: ${currentVersion}, Latest version: ${latestVersion}`);
        
        // Simple version comparison (assumes semantic versioning)
        if (isNewerVersion(latestVersion, currentVersion)) {
            console.log('New version available:', release.name);
            return release;
        }
        
        console.log('Already up to date');
        return null;
    } catch (error) {
        console.error('Failed to check for updates:', error);
        return null;
    }
}

function isNewerVersion(latest: string, current: string): boolean {
    const parseVersion = (version: string) => {
        return version.split('.').map(num => parseInt(num, 10));
    };
    
    const latestParts = parseVersion(latest);
    const currentParts = parseVersion(current);
    
    for (let i = 0; i < Math.max(latestParts.length, currentParts.length); i++) {
        const latestPart = latestParts[i] || 0;
        const currentPart = currentParts[i] || 0;
        
        if (latestPart > currentPart) return true;
        if (latestPart < currentPart) return false;
    }
    
    return false;
}

function showUpdateNotification(release: GitHubRelease) {
    if (!updateNotification || !updateVersionSpan || !updateButton) return;
    
    updateVersionSpan.textContent = `버전 ${release.tag_name}`;
    updateButton.onclick = () => {
        window.open(release.html_url, '_blank');
    };
    
    updateNotification.classList.remove('hidden');
    
    // Store the last check time
    localStorage.setItem('last_update_check', Date.now().toString());
    localStorage.setItem('latest_release_info', JSON.stringify(release));
}

function hideUpdateNotification() {
    if (updateNotification) {
        updateNotification.classList.add('hidden');
    }
}

async function initializeUpdateChecker() {
    if (!APP_CONFIG.checkForUpdates) return;
    
    // Check if we should check for updates
    const lastCheck = localStorage.getItem('last_update_check');
    const now = Date.now();
    
    if (lastCheck && (now - parseInt(lastCheck)) < (24 * 60 * 60 * 1000)) { // 24시간마다 체크
        console.log('Update check skipped (too recent)');
        
        // Show notification if we have a cached update
        const cachedRelease = localStorage.getItem('latest_release_info');
        if (cachedRelease) {
            try {
                const release = JSON.parse(cachedRelease);
                showUpdateNotification(release);
            } catch (error) {
                console.error('Failed to parse cached release info:', error);
            }
        }
        return;
    }
    
    // Check for updates
    const release = await checkForUpdates();
    if (release) {
        showUpdateNotification(release);
    } else {
        // Clear any cached update notification
        localStorage.removeItem('latest_release_info');
        hideUpdateNotification();
    }
}

async function manualUpdateCheck() {
    // Show loading state
    if (checkUpdateButton) {
        checkUpdateButton.disabled = true;
        checkUpdateButton.innerHTML = '<span>확인 중...</span>';
    }
    
    try {
        console.log('Manual update check initiated');
        const release = await checkForUpdates();
        
        if (release) {
            showUpdateNotification(release);
            alert(`새 업데이트가 있습니다! 버전 ${release.tag_name}`);
        } else {
            alert('현재 최신 버전을 사용 중입니다.');
            hideUpdateNotification();
        }
        
        // Update last check time
        localStorage.setItem('last_update_check', Date.now().toString());
    } catch (error) {
        console.error('Manual update check failed:', error);
        alert('업데이트 확인 중 오류가 발생했습니다.');
    } finally {
        // Restore button state
        if (checkUpdateButton) {
            checkUpdateButton.disabled = false;
            checkUpdateButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"><path d="M480-160q-134 0-227-93t-93-227q0-134 93-227t227-93q69 0 132 28.5T720-690v-110h80v280H520v-80h168q-32-56-87.5-88T480-720q-100 0-170 70t-70 170q0 100 70 170t170 70q77 0 139-44t87-116h84q-28 106-114 173t-196 67Z"/></svg>
                <span>업데이트 확인</span>
            `;
        }
    }
}



function saveSecureAPIKeys(keys: SecureAPIKeys) {
    saveAPIKeysToStorage(keys);
}

function loadSecureAPIKeys(): SecureAPIKeys {
    console.log('🔑 [API Keys] Loading secure API keys...');
    
    // First priority: Use cached decrypted keys if available
    if (cachedAPIKeys) {
        console.log('🔑 [API Keys] Using cached decrypted keys:', Object.keys(cachedAPIKeys));
        console.log('🔑 [API Keys] Cached key values:', Object.fromEntries(
            Object.entries(cachedAPIKeys).map(([k, v]) => [k, v ? `${v.length} chars` : 'empty'])
        ));
        return cachedAPIKeys;
    }
    
    // Load from simple storage
    const keys = loadAPIKeysFromStorage();
    cachedAPIKeys = keys;
    console.log('🔑 [API Keys] Loaded API keys from storage:', Object.keys(keys));
    return keys;
}

function getAPIKey(provider: string): string {
    console.log(`[API Key] Requesting key for ${provider}`);
    
    // 1순위: UI에서 직접 입력한 API 키 (암호화되어 로컬 파일에 저장)
    const keys = loadSecureAPIKeys();
    console.log(`[API Key] Available UI keys:`, Object.keys(keys));
    
    const uiKey = keys[provider as keyof SecureAPIKeys];
    if (uiKey) {
        console.log(`[API Key] ✓ Using encrypted local key for ${provider}`, `(length: ${uiKey.length})`);
        return uiKey;
    }
    
    // 2순위: 환경변수 (.env 파일, 개발용)
    const envKey = (import.meta as any).env?.[`VITE_${provider.toUpperCase()}_API_KEY`];
    if (envKey) {
        console.log(`[API Key] ✓ Using environment key for ${provider}`, `(length: ${envKey.length})`);
        return envKey;
    }
    
    console.log(`[API Key] ❌ No key found for ${provider}`);
    return '';
}

// Simple API key storage (no complex encryption)
function saveAPIKeysToStorage(keys: SecureAPIKeys): void {
    try {
        const keyData = JSON.stringify(keys);
        localStorage.setItem('api_keys', keyData);
        console.log('API keys saved successfully');
    } catch (error) {
        console.error('Failed to save API keys:', error);
    }
}

function loadAPIKeysFromStorage(): SecureAPIKeys {
    try {
        const keyData = localStorage.getItem('api_keys');
        if (!keyData) return {};
        
        const keys = JSON.parse(keyData);
        console.log('API keys loaded successfully');
        return keys;
    } catch (error) {
        console.error('Failed to load API keys:', error);
        return {};
    }
}

// Modal Management Functions  
async function showAPISettingsModal() {
    if (!apiSettingsModal) return;
    
    // Show modal first
    apiSettingsModal.classList.remove('hidden');
    
    // Load existing keys from UI storage
    let keys = loadSecureAPIKeys();
    
    // If no cached keys and new format exists, try to load it
    if (Object.keys(keys).length === 0 && localStorage.getItem('encrypted_api_keys')) {
        try {
            console.log('Loading keys for modal...');
            keys = loadSecureAPIKeys();
        } catch (error) {
            console.log('Could not load encrypted keys, using fallback');
        }
    }
    
    // Helper function: If no UI key exists, pre-fill with .env values
    const getDisplayKey = (provider: string) => {
        const uiKey = keys[provider as keyof SecureAPIKeys];
        if (uiKey) return uiKey;
        
        // Pre-fill from .env if available
        const envKey = (import.meta as any).env?.[`VITE_${provider.toUpperCase()}_API_KEY`];
        return envKey || '';
    };
    
    if (geminiKeyInput) geminiKeyInput.value = getDisplayKey('gemini');
    if (openaiKeyInput) openaiKeyInput.value = getDisplayKey('openai');
    if (upstageKeyInput) upstageKeyInput.value = getDisplayKey('upstage');
    
    console.log('Modal populated with keys:', Object.keys(keys));
    document.body.style.overflow = 'hidden';
    
    // Focus first input
    if (geminiKeyInput) geminiKeyInput.focus();
}

function hideAPISettingsModal() {
    if (!apiSettingsModal) return;
    
    apiSettingsModal.classList.add('hidden');
    document.body.style.overflow = '';
    
    // Clear input values for security
    if (geminiKeyInput) geminiKeyInput.value = '';
    if (openaiKeyInput) openaiKeyInput.value = '';
    if (upstageKeyInput) upstageKeyInput.value = '';
}

async function saveAPIKeysFromModal() {
    console.log('💾 [API Keys] Saving API keys from modal...');
    const keys: SecureAPIKeys = {};
    
    if (geminiKeyInput?.value.trim()) {
        keys.gemini = geminiKeyInput.value.trim();
        console.log('💾 [API Keys] Gemini key added, length:', keys.gemini.length);
    }
    if (openaiKeyInput?.value.trim()) {
        keys.openai = openaiKeyInput.value.trim();
        console.log('💾 [API Keys] OpenAI key added, length:', keys.openai.length);
    }
    if (upstageKeyInput?.value.trim()) {
        keys.upstage = upstageKeyInput.value.trim();
        console.log('💾 [API Keys] Upstage key added, length:', keys.upstage.length);
    }
    
    console.log('💾 [API Keys] Total keys to save:', Object.keys(keys).length);
    
    try {
        // Save keys
        saveAPIKeysToStorage(keys);
        
        // Update cached keys immediately
        cachedAPIKeys = keys;
        
        hideAPISettingsModal();
        
        console.log('💾 [API Keys] Keys saved, updating provider status...');
        console.log('💾 [API Keys] Cached keys after save:', Object.keys(cachedAPIKeys || {}));
        
        // Force refresh provider status multiple times to ensure it sticks
        await updateProviderPillsStatus();
        setTimeout(async () => {
            console.log('💾 [API Keys] Second status update...');
            await updateProviderPillsStatus();
        }, 500);
        setTimeout(async () => {
            console.log('💾 [API Keys] Third status update...');
            await updateProviderPillsStatus();
        }, 1000);
        
        // Show success message
        alert('API 키가 안전하게 저장되었습니다.');
        
        console.log('💾 [API Keys] API keys saved and cached successfully:', Object.keys(keys));
    } catch (error) {
        console.error('Failed to save API keys:', error);
        alert('API 키 저장에 실패했습니다. 다시 시도해주세요.');
    }
}

// Initialize API keys from .env on first run
function initializeAPIKeysFromEnv() {
    const existingKeys = loadSecureAPIKeys();
    
    // Check if we already have some UI keys stored
    const hasUIKeys = Object.keys(existingKeys).length > 0;
    if (hasUIKeys) return; // Don't overwrite existing UI settings
    
    // Import .env keys to UI storage for first-time users
    const envKeys: SecureAPIKeys = {};
    const providers = ['gemini', 'openai', 'upstage'];
    
    providers.forEach(provider => {
        const envKey = (import.meta as any).env?.[`VITE_${provider.toUpperCase()}_API_KEY`];
        if (envKey && envKey.trim()) {
            envKeys[provider as keyof SecureAPIKeys] = envKey.trim();
        }
    });
    
    // Save to UI storage if we found any .env keys
    if (Object.keys(envKeys).length > 0) {
        console.log('Initializing API keys from .env file:', Object.keys(envKeys));
        saveSecureAPIKeys(envKeys);
        
        // Show user-friendly notification
        setTimeout(() => {
            alert(`API 키가 자동으로 설정되었습니다!\n\n설정된 제공자: ${Object.keys(envKeys).join(', ')}\n\n"API 설정" 버튼에서 키를 변경할 수 있습니다.`);
        }, 2000); // 2초 후 표시 (UI 로딩 완료 후)
    }
}

function togglePasswordVisibility(targetId: string) {
    const input = document.getElementById(targetId) as HTMLInputElement;
    if (!input) return;
    
    const isPassword = input.type === 'password';
    input.type = isPassword ? 'text' : 'password';
    
    // Update icon (you could implement icon change logic here)
}

// Initialize UI after DOM elements are available
function initializeUI() {
    // Initialize API keys from .env on first run
    initializeAPIKeysFromEnv();
    
    updateProviderPillsStatus();
    updateProcessButtonState();
}

// Update process button state based on settings and file selection
function updateProcessButtonState() {
    const hasApiKey = isProviderAvailableSync(currentSettings.provider);
    const hasSelectedPages = selectedPages.length > 0;
    console.log('updateProcessButtonState:', { provider: currentSettings.provider, hasApiKey, selectedPagesCount: selectedPages.length });
    processButton.disabled = !hasApiKey || !hasSelectedPages;
    
    // Update button text based on availability
    if (!hasApiKey && hasSelectedPages) {
        processButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="m370-80-16-128q-13-5-24.5-12T307-235l-119 50L78-375l103-78q-1-7-1-13.5v-27q0-6.5 1-13.5L78-585l110-190 119 50q11-8 23-15t24-12l16-128h220l16 128q13 5 24.5 12t22.5 15l119-50 110 190-103 78q1 7 1 13.5v27q0 6.5-1 13.5l103 78-110 190-119-50q-11 8-23 15t-24 12L590-80H370Z"/></svg><span>API 키 설정 필요</span>`;
    } else if (hasApiKey && !hasSelectedPages) {
        processButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="M480-80q-84 0-158-30.5T195-195q-54-54-84.5-128T80-480q0-84 30.5-158T195-765q54-54 128-84.5T480-880q52 0 101 12.5T673-831l-47 47q-40-10-86-10-142 0-241 99t-99 241q0 142 99 241t241 99q116 0 211.5-73T774-480h-84l152-152 152 152h-84q-22 134-118 221t-226 87Z"/></svg><span>파일 선택 필요</span>`;
    } else if (hasApiKey && hasSelectedPages) {
        processButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="M480-80q-84 0-158-30.5T195-195q-54-54-84.5-128T80-480q0-84 30.5-158T195-765q54-54 128-84.5T480-880q52 0 101 12.5T673-831l-47 47q-40-10-86-10-142 0-241 99t-99 241q0 142 99 241t241 99q116 0 211.5-73T774-480h-84l152-152 152 152h-84q-22 134-118 221t-226 87Z"/></svg><span>데이터 추출</span>`;
    } else {
        processButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="m370-80-16-128q-13-5-24.5-12T307-235l-119 50L78-375l103-78q-1-7-1-13.5v-27q0-6.5 1-13.5L78-585l110-190 119 50q11-8 23-15t24-12l16-128h220l16 128q13 5 24.5 12t22.5 15l119-50 110 190-103 78q1 7 1 13.5v27q0 6.5-1 13.5l103 78-110 190-119-50q-11 8-23 15t-24 12L590-80H370Z"/></svg><span>API 키 및 파일 필요</span>`;
    }
}

// Functions
function renderTable() {
    if (!tableBody) return;
    tableBody.innerHTML = '';

    tableData.forEach(rowData => {
        const row = document.createElement('tr');
        row.dataset.id = rowData.id.toString();

        row.innerHTML = `
            <td><input type="text" value="${rowData.date}" data-field="date" title="${rowData.date}"></td>
            <td><input type="text" value="${rowData.quantity}" data-field="quantity" title="${rowData.quantity}"></td>
            <td><input type="text" value="${rowData.amountUSD}" data-field="amountUSD" title="${rowData.amountUSD}"></td>
            <td><input type="text" value="${rowData.commissionUSD}" data-field="commissionUSD" title="${rowData.commissionUSD}"></td>
            <td><input type="text" value="${rowData.totalUSD}" data-field="totalUSD" title="${rowData.totalUSD}"></td>
            <td><input type="text" value="${rowData.totalKRW}" data-field="totalKRW" title="${rowData.totalKRW}"></td>
            <td><input type="text" value="${rowData.balanceKRW}" data-field="balanceKRW" title="${rowData.balanceKRW}"></td>
            <td>
                <button class="delete-button" aria-label="행 삭제">
                    <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"><path d="M280-120q-33 0-56.5-23.5T200-200v-520h-40v-80h200v-40h240v40h200v80h-40v520q0 33-23.5 56.5T680-120H280Zm400-600H280v520h400v-520ZM360-280h80v-360h-80v360Zm160 0h80v-360h-80v360ZM280-720v520-520Z"/></svg>
                </button>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

function addRow(data: Partial<Omit<TableRowData, 'id'>> = {}) {
    const newRow: TableRowData = {
        id: nextId++,
        date: data.date || '',
        quantity: data.quantity || '',
        amountUSD: data.amountUSD || '',
        commissionUSD: data.commissionUSD || '',
        totalUSD: data.totalUSD || '',
        totalKRW: data.totalKRW || '',
        balanceKRW: data.balanceKRW || '',
    };
    tableData.push(newRow);
    renderTable();
    setupInputTooltips();
}

// Setup tooltips for input fields
function setupInputTooltips() {
    const inputs = tableBody.querySelectorAll('input[title]');
    inputs.forEach(input => {
        const inputElement = input as HTMLInputElement;
        // Ensure tooltip shows current value
        inputElement.title = inputElement.value;
    });
}

function deleteRow(id: number) {
    tableData = tableData.filter(row => row.id !== id);
    renderTable();
}

function updateCell(id: number, field: keyof Omit<TableRowData, 'id'>, value: string) {
    const rowIndex = tableData.findIndex(row => row.id === id);
    if (rowIndex > -1) {
        (tableData[rowIndex] as any)[field] = value;
    }
}

async function copyTableToClipboard() {
    if (tableData.length === 0) {
        alert('복사할 데이터가 없습니다.');
        return;
    }

    const originalButtonText = copyButton.innerHTML;
    copyButton.disabled = true;

    try {
        const rows = tableData.map(rowData => {
            const cleanNumber = (value: string | number) => String(value).replace(/,/g, '');
            const excelRow = [
                '',
                rowData.date,
                cleanNumber(rowData.quantity),
                cleanNumber(rowData.amountUSD),
                cleanNumber(rowData.commissionUSD),
                cleanNumber(rowData.totalUSD),
                cleanNumber(rowData.totalKRW),
                '',
                cleanNumber(rowData.balanceKRW),
                ''
            ];
            return excelRow.join('\t');
        }).join('\n');

        await navigator.clipboard.writeText(rows);
        
        const successIcon = `<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="m420-200 280-280-56-56-224 224-114-114-56 56 170 170Zm60-280q-83 0-141.5-58.5T280-680q0-83 58.5-141.5T480-880q83 0 141.5 58.5T680-680q0 83-58.5 141.5T480-480Z"/></svg>`;
        copyButton.innerHTML = `${successIcon}<span>복사 완료!</span>`;

        setTimeout(() => {
            copyButton.innerHTML = originalButtonText;
            copyButton.disabled = false;
        }, 2000);

    } catch (err) {
        console.error('클립보드 복사 실패:', err);
        alert('클립보드에 복사하지 못했습니다.');
        copyButton.innerHTML = originalButtonText;
        copyButton.disabled = false;
    }
}

function resetUploadUI() {
    selectedPages = [];
    pdfFileGroups.clear();
    imagePreview.src = '';
    imagePreview.classList.add('hidden');
    dropZone.querySelector('p')?.classList.remove('hidden');
    pdfPreviewContainer.classList.add('hidden');
    if (pdfPagesContainer) pdfPagesContainer.innerHTML = '';
    updateProcessButtonState();
}

function renderPdfPages() {
    if (!pdfPagesContainer) return;
    
    pdfPagesContainer.innerHTML = '';
    
    // Show or hide PDF preview container based on whether we have files
    if (pdfFileGroups.size > 0) {
        pdfPreviewContainer.classList.remove('hidden');
    } else {
        pdfPreviewContainer.classList.add('hidden');
        return; // No files to render
    }
    
    // Group pages by filename and render
    for (const [fileName, pages] of pdfFileGroups.entries()) {
        // Create file group container
        const fileGroup = document.createElement('div');
        fileGroup.className = 'pdf-file-group';
        
        // Create file header
        const fileHeader = document.createElement('div');
        fileHeader.className = 'pdf-file-header';
        
        const fileNameDiv = document.createElement('div');
        fileNameDiv.className = 'pdf-file-name';
        fileNameDiv.textContent = fileName;
        
        const pageCountDiv = document.createElement('div');
        pageCountDiv.className = 'pdf-page-count';
        const selectedCount = pages.filter(page => selectedPages.some(sp => sp.data === page.data)).length;
        pageCountDiv.textContent = `${selectedCount}/${pages.length} 페이지 선택됨`;
        
        fileHeader.appendChild(fileNameDiv);
        fileHeader.appendChild(pageCountDiv);
        fileGroup.appendChild(fileHeader);
        
        // Create pages container for this file
        const filePagesContainer = document.createElement('div');
        filePagesContainer.className = 'pdf-file-pages';
        
        // Sort pages by page number
        const sortedPages = [...pages].sort((a, b) => (a.pageNumber || 0) - (b.pageNumber || 0));
        
        sortedPages.forEach(pageData => {
            // Create page item container
            const pageItem = document.createElement('div');
            pageItem.className = 'pdf-page-item';
            pageItem.setAttribute('data-page-id', pageData.data.substring(0, 20));
            
            // Create thumbnail
            const img = document.createElement('img');
            img.src = `data:${pageData.mimeType};base64,${pageData.data}`;
            img.className = 'pdf-page-thumbnail';
            img.alt = `${fileName} ${pageData.pageNumber}페이지`;
            
            // Check if this page is selected
            if (selectedPages.some(sp => sp.data === pageData.data)) {
                img.classList.add('selected');
            }
            
            // Create page label
            const label = document.createElement('div');
            label.className = 'pdf-page-label';
            label.textContent = `페이지 ${pageData.pageNumber}`;
            
            // Create remove button
            const removeBtn = document.createElement('button');
            removeBtn.className = 'pdf-page-remove';
            removeBtn.innerHTML = '×';
            removeBtn.title = '페이지 제거';
            
            // Add click event for thumbnail
            img.addEventListener('click', () => {
                img.classList.toggle('selected');
                
                if (img.classList.contains('selected')) {
                    selectedPages.push(pageData);
                    imagePreview.src = img.src;
                    imagePreview.classList.remove('hidden');
                    dropZone.querySelector('p')?.classList.add('hidden');
                } else {
                    selectedPages = selectedPages.filter(p => p.data !== pageData.data);
                    if (selectedPages.length === 0) {
                        imagePreview.classList.add('hidden');
                        dropZone.querySelector('p')?.classList.remove('hidden');
                    }
                }
                
                // Update page count
                const selectedCount = pages.filter(page => selectedPages.some(sp => sp.data === page.data)).length;
                pageCountDiv.textContent = `${selectedCount}/${pages.length} 페이지 선택됨`;
                
                updateProcessButtonState();
            });
            
            // Add click event for remove button
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                removePage(pageData);
            });
            
            pageItem.appendChild(img);
            pageItem.appendChild(label);
            pageItem.appendChild(removeBtn);
            filePagesContainer.appendChild(pageItem);
        });
        
        fileGroup.appendChild(filePagesContainer);
        pdfPagesContainer.appendChild(fileGroup);
    }
}


function removePage(pageData: PageData) {
    // Add to delete history for undo
    deleteHistory.push({
        pageData: { ...pageData }, // Deep copy
        timestamp: Date.now()
    });
    
    // Keep only recent delete actions
    if (deleteHistory.length > MAX_UNDO_HISTORY) {
        deleteHistory.shift();
    }
    
    // Remove from selected pages
    selectedPages = selectedPages.filter(p => p.data !== pageData.data);
    
    // Remove from file groups
    const fileName = pageData.fileName;
    const pages = pdfFileGroups.get(fileName);
    if (pages) {
        const updatedPages = pages.filter(p => p.data !== pageData.data);
        if (updatedPages.length === 0) {
            pdfFileGroups.delete(fileName);
        } else {
            pdfFileGroups.set(fileName, updatedPages);
        }
    }
    
    // Update UI with fade out animation
    const pageElement = document.querySelector(`[data-page-id="${pageData.data.substring(0, 20)}"]`) as HTMLElement;
    if (pageElement) {
        pageElement.style.transition = 'all 0.3s ease-out';
        pageElement.style.opacity = '0';
        pageElement.style.transform = 'scale(0.8) translateY(-20px)';
        
        setTimeout(() => {
            renderPdfPages();
        }, 300);
    } else {
        renderPdfPages();
    }
    
    // Update preview if needed
    if (selectedPages.length === 0) {
        imagePreview.classList.add('hidden');
        dropZone.querySelector('p')?.classList.remove('hidden');
    }
    
    // renderPdfPages will handle showing/hiding the container
    
    updateProcessButtonState();
    
    // Show undo notification
    showUndoNotification();
}

function undoLastDelete() {
    console.log('undoLastDelete called, deleteHistory.length:', deleteHistory.length);
    if (deleteHistory.length === 0) return;
    
    const lastDelete = deleteHistory.pop();
    if (!lastDelete) return;
    
    console.log('Restoring deleted page:', lastDelete.pageData.fileName);
    
    const { pageData } = lastDelete;
    const fileName = pageData.fileName;
    
    // Add back to file groups
    if (pdfFileGroups.has(fileName)) {
        const pages = pdfFileGroups.get(fileName)!;
        pages.push(pageData);
        // Sort pages by page number
        pages.sort((a, b) => (a.pageNumber || 0) - (b.pageNumber || 0));
    } else {
        pdfFileGroups.set(fileName, [pageData]);
    }
    
    // Update UI
    renderPdfPages();
    
    updateProcessButtonState();
    
    // Hide undo notification if no more items
    if (deleteHistory.length === 0) {
        hideUndoNotification();
    }
}

function showUndoNotification() {
    // Remove existing notification
    const existing = document.querySelector('.undo-notification');
    if (existing) existing.remove();
    
    const notification = document.createElement('div');
    notification.className = 'undo-notification';
    notification.innerHTML = `
        <div class="undo-content">
            <span>페이지가 삭제되었습니다</span>
            <button id="undo-button">실행취소 (Ctrl+Z)</button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Add click event
    notification.querySelector('#undo-button')?.addEventListener('click', undoLastDelete);
    
    // Auto hide after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

function hideUndoNotification() {
    const notification = document.querySelector('.undo-notification');
    if (notification) {
        notification.classList.add('fade-out');
        setTimeout(() => notification.remove(), 300);
    }
}

function handleImageFile(file: File) {
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const result = e.target?.result as string;
            
            // 이미지 데이터 유효성 검사
            if (!result || typeof result !== 'string') {
                console.error(`이미지 파일 ${file.name} 읽기 실패: 빈 결과`);
                alert(`이미지 파일 ${file.name}을 읽을 수 없습니다.`);
                return;
            }
            
            if (!result.includes(',')) {
                console.error(`이미지 파일 ${file.name} 처리 실패: 잘못된 데이터 URL 형식`);
                alert(`이미지 파일 ${file.name}의 형식이 올바르지 않습니다.`);
                return;
            }
            
            const base64Data = result.split(',')[1];
            if (!base64Data || base64Data.length === 0) {
                console.error(`이미지 파일 ${file.name} 처리 실패: 빈 base64 데이터`);
                alert(`이미지 파일 ${file.name}에서 데이터를 추출할 수 없습니다.`);
                return;
            }
            
            // MIME 타입 유효성 검사
            const mimeType = file.type || 'image/png';
            if (!mimeType.startsWith('image/')) {
                console.error(`이미지 파일 ${file.name} 처리 실패: 잘못된 MIME 타입 ${mimeType}`);
                alert(`파일 ${file.name}은 이미지 파일이 아닙니다.`);
                return;
            }
            
            const pageData: PageData = {
                data: base64Data,
                mimeType: mimeType,
                fileName: file.name || 'unknown.png'
            };
            
            console.log(`✅ 이미지 파일 처리 완료:`, {
                fileName: pageData.fileName,
                mimeType: pageData.mimeType,
                dataLength: pageData.data.length
            });
        
        // Add to file groups - if file already exists, replace it
        if (pdfFileGroups.has(file.name)) {
            // Remove any existing pages from selected pages
            const existingPages = pdfFileGroups.get(file.name) || [];
            existingPages.forEach(existingPage => {
                selectedPages = selectedPages.filter(p => p.data !== existingPage.data);
            });
        }
        pdfFileGroups.set(file.name, [pageData]);
        
        // Auto-select the uploaded page
            selectedPages.push(pageData);
            
            // Show preview
            imagePreview.src = result;
            imagePreview.classList.remove('hidden');
            dropZone.querySelector('p')?.classList.add('hidden');
            
            // Render PDF pages to show the image file too
            renderPdfPages();
            
            // Update button state after page is selected
            updateProcessButtonState();
        } catch (error) {
            console.error(`이미지 파일 ${file.name} 처리 중 오류:`, error);
            alert(`이미지 파일 ${file.name} 처리 중 오류가 발생했습니다: ${error instanceof Error ? error.message : String(error)}`);
        }
    };
    
    reader.onerror = () => {
        console.error(`이미지 파일 ${file.name} 읽기 실패`);
        alert(`이미지 파일 ${file.name}을 읽는 중 오류가 발생했습니다.`);
    };
    
    reader.readAsDataURL(file);
}

async function handlePdfFile(file: File) {
    const reader = new FileReader();
    reader.onload = async (e) => {
        const typedarray = new Uint8Array(e.target?.result as ArrayBuffer);
        
        loadingOverlay.classList.remove('hidden');
        loadingOverlay.querySelector('p')!.textContent = 'PDF 파일을 읽는 중입니다...';

        try {
            const pdf = await pdfjsLib.getDocument(typedarray).promise;
            
            const pages: PageData[] = [];

            const numPages = pdf.numPages;
            for (let i = 1; i <= numPages; i++) {
                const page = await pdf.getPage(i);
                const viewport = page.getViewport({ scale: 1.5 });
                
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;

                await page.render({ canvasContext: context!, viewport: viewport }).promise;
                
                const pageDataUrl = canvas.toDataURL('image/png');
                
                // 페이지 데이터 유효성 검사
                if (!pageDataUrl || !pageDataUrl.includes(',')) {
                    console.error(`PDF 페이지 ${i} 렌더링 실패: 빈 데이터 URL`);
                    continue;
                }
                
                const base64Data = pageDataUrl.split(',')[1];
                if (!base64Data || base64Data.length === 0) {
                    console.error(`PDF 페이지 ${i} 렌더링 실패: 빈 base64 데이터`);
                    continue;
                }
                
                const pageInfo: PageData = {
                    data: base64Data,
                    mimeType: 'image/png',
                    fileName: file.name || `page-${i}.pdf`,
                    pageNumber: i
                };
                pages.push(pageInfo);
            }
            
            // Add pages to file groups - if file already exists, replace it
            if (pdfFileGroups.has(file.name)) {
                // Remove any existing pages from selected pages
                const existingPages = pdfFileGroups.get(file.name) || [];
                existingPages.forEach(existingPage => {
                    selectedPages = selectedPages.filter(p => p.data !== existingPage.data);
                });
            }
            pdfFileGroups.set(file.name, pages);
            
            // Auto-select all uploaded pages
            pages.forEach(page => {
                selectedPages.push(page);
            });
            
            // Render the updated PDF pages
            renderPdfPages();
            
            // Update button state after pages are selected
            updateProcessButtonState();
        } catch (error) {
            console.error("Error processing PDF:", error);
            alert(`PDF 처리 중 오류가 발생했습니다: ${error instanceof Error ? error.message : String(error)}`);
            resetUploadUI();
        } finally {
            loadingOverlay.classList.add('hidden');
            loadingOverlay.querySelector('p')!.textContent = 'AI가 문서를 분석 중입니다...';
        }
    };
    reader.readAsArrayBuffer(file);
}

function handleFilesSelect(files: FileList | File[]) {
    if (!files || files.length === 0) return;

    // Don't reset UI if we're adding more files
    if (selectedPages.length === 0 && pdfFileGroups.size === 0) {
        resetUploadUI();
    }

    // Process each file
    Array.from(files).forEach(file => {
        if (file.type.startsWith('image/')) {
            handleImageFile(file);
        } else if (file.type === 'application/pdf') {
            handlePdfFile(file);
        } else {
            alert(`${file.name}: 이미지 또는 PDF 파일만 업로드할 수 있습니다.`);
        }
    });
}

// Keep backward compatibility
function handleFileSelect(file: File) {
    handleFilesSelect([file]);
}

// --- 추가된 코드 시작 ---
// 로그 파일 다운로드 함수
async function logToFile(content: string, filename: string) {
    try {
        // 파일 이름에 타임스탬프를 추가하여 겹치지 않게 함
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const fullFilename = `${timestamp}-${filename}`;

        console.log(`--- 로그: ${fullFilename} ---`);
        console.log(content);
        console.log(`--- 로그 끝: ${fullFilename} ---`);

        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fullFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (e) {
        console.error(`${filename} 파일에 로그를 기록하지 못했습니다:`, e);
    }
}
// --- 추가된 코드 끝 ---

// Data validation and cleaning function
function validateAndCleanExtractedData(data: any) {
    console.log('Raw AI response data:', data);
    
    // Ensure all required fields exist with default values
    const cleanedData = {
        date: data.date || '',
        quantity: parseFloat(data.quantity) || 0,
        amountUSD: parseFloat(data.amountUSD) || 0,
        commissionUSD: parseFloat(data.commissionUSD) || 0,
        totalUSD: parseFloat(data.totalUSD) || 0,
        totalKRW: parseFloat(data.totalKRW) || 0,
        balanceKRW: parseFloat(data.balanceKRW) || 0
    };
    
    console.log('Cleaned data:', cleanedData);
    return cleanedData;
}

// AI Processing Functions
async function processWithGemini(pageData: PageData) {
    console.log('🚀 [Gemini] Starting Gemini processing with model:', currentSettings.model);
    console.log('🚀 [Gemini] Platform:', navigator.platform, 'UserAgent:', navigator.userAgent.substring(0, 50));
    
    const textPart = {
        text: "제공된 수입 정산서 문서에서 정확한 항목별로 데이터를 추출해 주세요:\n\n1. date: 문서의 작성일 (YYYY-MM-DD 형식)\n2. quantity: 수량 (GT 단위)\n3. amountUSD: COMMERCIAL INVOICE CHARGE의 US$ 금액\n4. commissionUSD: COMMISSION의 US$ 금액\n5. totalUSD: '입금하신 금액' 또는 '수수료포함금액'의 US$ 금액 (총 경비가 아님)\n6. totalKRW: '입금하신 금액' 또는 '수수료포함금액'의 원화(₩) 금액 (총 경비가 아님)\n7. balanceKRW: 잔액의 원화(₩) 금액\n\n주의사항: totalUSD와 totalKRW는 반드시 '입금하신 금액' 섹션에서 추출하세요."
    };

    const imagePart = {
        inlineData: { mimeType: pageData.mimeType, data: pageData.data },
    };

    try {
        console.log('🚀 [Gemini] Checking API key...');
        const apiKey = getAPIKey('gemini');
        if (!apiKey || apiKey.trim() === '') {
            console.error('❌ [Gemini] API key not found or empty');
            throw new Error('Gemini API key not configured. Please set your API key in the settings.');
        }
        console.log('✅ [Gemini] API key found, length:', apiKey.length);

        console.log('🚀 [Gemini] Sending request through proxy server...');
        console.log('🚀 [Gemini] Proxy URL: http://localhost:3002/api/gemini');
        console.log('🚀 [Gemini] Image data length:', pageData.data.length, 'MIME type:', pageData.mimeType);
        
        const response = await fetch('http://localhost:3002/api/gemini', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                apiKey: apiKey, // UI에서 입력한 API 키 전달
                model: currentSettings.model,
                contents: [{ parts: [textPart, imagePart] }],
                config: {
                    responseMimeType: "application/json",
                }
            })
        });

        console.log('✅ [Gemini] Response received, status:', response.status, response.statusText);
        console.log('✅ [Gemini] Response headers:', Object.fromEntries(response.headers.entries()));

        if (!response.ok) {
            console.error('❌ [Gemini] Proxy API request failed with status:', response.status);
            
            // Fallback to direct API call if proxy fails
            if (response.status >= 500 || response.status === 0) {
                console.warn('⚠️ [Gemini] Proxy server error, trying direct API call...');
                
                const directResponse = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${currentSettings.model}:generateContent?key=${apiKey}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        contents: [{ parts: [textPart, imagePart] }],
                        generationConfig: {
                            responseMimeType: "application/json",
                        }
                    })
                });
                
                if (directResponse.ok) {
                    console.log('✅ [Gemini] Direct API call successful');
                    const result = await directResponse.json();
                    const jsonText = result.candidates[0].content.parts[0].text;
                    const parsedData = JSON.parse(jsonText);
                    return validateAndCleanExtractedData(parsedData);
                }
            }
            
            const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
            console.error('Gemini proxy error:', errorData);
            throw new Error(`Gemini API 오류 (${response.status}): ${errorData.error || response.statusText}`);
        }

        const result = await response.json();
        console.log('Gemini raw response:', result);
        
        const jsonText = result.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || '';
        console.log('Extracted JSON text:', jsonText);
        
        if (!jsonText) {
            throw new Error('AI 응답이 비어있습니다.');
        }
        
        const parsedData = JSON.parse(jsonText);
        return validateAndCleanExtractedData(parsedData);
        
    } catch (error) {
        console.error('❌ [Gemini] Processing error:', error);
        console.error('❌ [Gemini] Error type:', error.constructor.name);
        console.error('❌ [Gemini] Error stack:', error.stack);
        console.error('❌ [Gemini] Browser:', navigator.userAgent);
        
        // Windows-specific debugging
        if (error instanceof Error) {
            if (error.message.includes('fetch')) {
                console.error('❌ [Gemini] Network error detected - possible CORS or firewall issue');
                throw new Error('네트워크 오류: CORS 정책이나 방화벽이 API 요청을 차단했을 수 있습니다.');
            }
            if (error.message.includes('Failed to fetch')) {
                console.error('❌ [Gemini] Failed to fetch - possible network connectivity issue');
                throw new Error('네트워크 연결 오류: 인터넷 연결이나 DNS 설정을 확인하세요.');
            }
        }
        
        throw error;
    }
}

async function processWithOpenAI(pageData: PageData) {
    console.log('🚀 [OpenAI] Starting OpenAI processing...');
    
    const apiKey = getAPIKey('openai');
    if (!apiKey || apiKey.trim() === '') {
        console.error('❌ [OpenAI] API key not found or empty');
        throw new Error('OpenAI API key not configured. Please set your API key in the settings.');
    }
    console.log('✅ [OpenAI] API key found, length:', apiKey.length);

    console.log('🚀 [OpenAI] Sending request through proxy server...');
    console.log('🚀 [OpenAI] Proxy URL: http://localhost:3002/api/openai');
    
    const response = await fetch('http://localhost:3002/api/openai', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            apiKey: apiKey, // UI에서 입력한 API 키 전달
            model: currentSettings.model,
            messages: [
                {
                    role: "user",
                    content: [
                        {
                            type: "text",
                            text: "제공된 수입 정산서 문서에서 정확한 항목별로 데이터를 추출해 주세요:\n\n1. date: 문서의 작성일 (YYYY-MM-DD 형식)\n2. quantity: 수량 (GT 단위)\n3. amountUSD: COMMERCIAL INVOICE CHARGE의 US$ 금액\n4. commissionUSD: COMMISSION의 US$ 금액\n5. totalUSD: '입금하신 금액' 또는 '수수료포함금액'의 US$ 금액 (총 경비가 아님)\n6. totalKRW: '입금하신 금액' 또는 '수수료포함금액'의 원화(₩) 금액 (총 경비가 아님)\n7. balanceKRW: 잔액의 원화(₩) 금액\n\n주의사항:\n- totalUSD와 totalKRW는 반드시 '입금하신 금액' 섹션에서 추출하세요\n- '총 경비' 항목이 아닌 '입금하신 금액' 또는 '수수료포함금액' 항목을 사용하세요\n\nJSON 형식으로 반환: {\"date\": \"YYYY-MM-DD\", \"quantity\": 숫자, \"amountUSD\": 숫자, \"commissionUSD\": 숫자, \"totalUSD\": 숫자, \"totalKRW\": 숫자, \"balanceKRW\": 숫자}"
                        },
                        {
                            type: "image_url",
                            image_url: {
                                url: `data:${pageData.mimeType};base64,${pageData.data}`
                            }
                        }
                    ]
                }
            ],
                response_format: { type: "json_object" }
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(`OpenAI API 오류: ${response.status} - ${errorData.details || errorData.error}`);
        }

    const result = await response.json();
    const content = result.choices?.[0]?.message?.content;
    if (!content) throw new Error('No response from OpenAI');
    const parsedData = JSON.parse(content);
    return validateAndCleanExtractedData(parsedData);
}


async function processWithUpstage(pageData: PageData) {
    console.log('🚀 [Upstage] Starting Upstage processing...');
    
    const apiKey = getAPIKey('upstage');
    if (!apiKey || apiKey.trim() === '') {
        console.error('❌ [Upstage] API key not found or empty');
        throw new Error('Upstage API key not configured. Please set your API key in the settings.');
    }
    console.log('✅ [Upstage] API key found, length:', apiKey.length);

    console.log('🚀 [Upstage] Sending request through proxy server...');
    console.log('🚀 [Upstage] Proxy URL: http://localhost:3002/api/upstage');
    
    const response = await fetch('http://localhost:3002/api/upstage', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            apiKey: apiKey, // UI에서 입력한 API 키 전달
            document: `data:${pageData.mimeType};base64,${pageData.data}`,
            include_tables: true // Optional parameter for table extraction
        })
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Upstage API 오류: ${response.status} - ${errorData.details || errorData.error}`);
    }

    const result = await response.json();
    
    // Parse the response from Upstage Document Parse API (Real format)
    try {
        console.log('🔍 [Upstage] Raw API response:', result);
        
        // Real Upstage API response format: result.content.text
        let extractedText = '';
        
        if (result.content && result.content.text) {
            // This is the correct format from real API response
            extractedText = result.content.text;
        } else if (result.text) {
            // Fallback for simplified responses
            extractedText = result.text;
        } else {
            // Log the full response structure to understand the format
            console.log('🔍 [Upstage] Full response structure:', JSON.stringify(result, null, 2));
            throw new Error('Upstage API 응답에서 텍스트를 찾을 수 없습니다. Expected: result.content.text');
        }
        
        console.log('📄 [Upstage] Extracted text length:', extractedText.length);
        console.log('📄 [Upstage] Extracted text preview:', extractedText.substring(0, 200) + '...');
        
        // Enhanced pattern matching for Korean import settlement documents
        const extractedData = {
            date: extractDateFromText(extractedText),
            quantity: extractNumberFromText(extractedText, ['수 량', '수량', 'GT']),
            amountUSD: extractNumberFromText(extractedText, ['COMMERCIAL INVOICE CARGE', 'COMMERCIAL INVOICE CHARGE', 'US$22,234.42', 'US$']),
            commissionUSD: extractNumberFromText(extractedText, ['COMMISSION', '수수료', 'US$222.34']),
            totalUSD: extractNumberFromText(extractedText, ['외화', '$22,456.76', 'US$22,456.76']),
            totalKRW: extractNumberFromText(extractedText, ['입금하신 금액', '₩33,072,070']),
            balanceKRW: extractNumberFromText(extractedText, ['잔 액', '잔액', '₩4,796,651'])
        };
        
        console.log('🔍 [Upstage] Extracted data:', extractedData);
        return validateAndCleanExtractedData(extractedData);
    } catch (parseError) {
        console.error('Upstage 응답 파싱 오류:', parseError);
        throw new Error('Upstage 응답을 파싱할 수 없습니다.');
    }
}

// Helper functions for text extraction
function extractDateFromText(text: string): string {
    const datePatterns = [
        /(\d{4}[-/]\d{1,2}[-/]\d{1,2})/,
        /(\d{1,2}[-/]\d{1,2}[-/]\d{4})/,
        /(\d{4}\.\d{1,2}\.\d{1,2})/,
        /(\d{1,2}\.\d{1,2}\.\d{4})/
    ];
    
    for (const pattern of datePatterns) {
        const match = text.match(pattern);
        if (match) {
            return match[1];
        }
    }
    return '';
}

function extractNumberFromText(text: string, keywords: string[]): number {
    for (const keyword of keywords) {
        const pattern = new RegExp(`${keyword}[^\\d]*([\\d,]+(?:\\.\\d+)?)`, 'i');
        const match = text.match(pattern);
        if (match) {
            const numberStr = match[1].replace(/,/g, '');
            const number = parseFloat(numberStr);
            if (!isNaN(number)) {
                return number;
            }
        }
    }
    return 0;
}



async function processDocument() {
    console.log('=== 데이터 추출 시작 ===');
    console.log('선택된 페이지 수:', selectedPages.length);
    console.log('현재 설정:', currentSettings);
    
    // selectedPages 배열 상세 검사
    console.log('📄 [Process] selectedPages 상세 정보:');
    selectedPages.forEach((page, index) => {
        console.log(`📄 [Process] Page ${index + 1}:`, {
            fileName: page?.fileName || 'unknown',
            mimeType: page?.mimeType || 'missing',
            dataLength: page?.data?.length || 0,
            pageNumber: page?.pageNumber || 'N/A',
            hasData: !!page?.data,
            hasMimeType: !!page?.mimeType
        });
    });
    
    if (selectedPages.length === 0) {
        alert('먼저 이미지나 PDF 페이지를 선택하세요.');
        return;
    }

    if (!isProviderAvailableSync(currentSettings.provider)) {
        alert(`${currentSettings.provider.toUpperCase()} 서비스에 연결할 수 없습니다. 설정을 확인해주세요.`);
        return;
    }

    loadingOverlay.classList.remove('hidden');
    processButton.disabled = true;
    hideRatingSection();

    const totalToProcess = selectedPages.length;
    let successCount = 0;
    const allExtractedData = [];
    
    try {
    
    // Start logging
    const startTime = Date.now();
    const logId = startLogging(currentSettings.provider, currentSettings.model, totalToProcess);

    for (let i = 0; i < totalToProcess; i++) {
        const page = selectedPages[i];
        loadingOverlay.querySelector('p')!.textContent = `AI가 문서를 분석 중입니다... (${i + 1}/${totalToProcess})`;

        try {
            // 페이지 데이터 유효성 검사
            if (!page) {
                throw new Error(`페이지 ${i + 1}: 페이지 데이터가 없습니다.`);
            }
            
            if (!page.data) {
                throw new Error(`페이지 ${i + 1}: 이미지 데이터가 없습니다.`);
            }
            
            if (!page.mimeType) {
                throw new Error(`페이지 ${i + 1}: MIME 타입 정보가 없습니다.`);
            }
            
            console.log(`✅ Processing page ${i + 1}:`, {
                provider: currentSettings.provider,
                fileName: page.fileName || 'unknown',
                mimeType: page.mimeType,
                dataLength: page.data.length,
                pageNumber: page.pageNumber || 'N/A'
            });
            
            let extractedData;
            
            switch (currentSettings.provider) {
                case 'gemini':
                    extractedData = await processWithGemini(page);
                    break;
                case 'openai':
                    extractedData = await processWithOpenAI(page);
                    break;
                case 'upstage':
                    extractedData = await processWithUpstage(page);
                    break;
                default:
                    throw new Error('지원되지 않는 AI 제공자입니다.');
            }
            
            console.log(`Successfully processed page ${i + 1}:`, extractedData);

            allExtractedData.push(extractedData);
            successCount++;

        } catch (error) {
            console.error(`Error processing page ${i + 1}:`, error);
            alert(`페이지 ${i + 1} 처리 중 오류가 발생했습니다: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
    
    allExtractedData.forEach((extractedData, index) => {
        console.log(`Formatting data for row ${index + 1}:`, extractedData);
        
        // Safe number formatting function
        const safeToLocaleString = (value: any, locale: string = 'ko-KR'): string => {
            if (value === null || value === undefined || value === '') {
                return '0';
            }
            
            // Convert to number if it's a string
            const numValue = typeof value === 'string' ? parseFloat(value.replace(/,/g, '')) : value;
            
            if (isNaN(numValue)) {
                console.warn(`Invalid number value: ${value}`);
                return '0';
            }
            
            return numValue.toLocaleString(locale);
        };
        
        const formattedData = {
             date: extractedData.date || '',
             quantity: safeToLocaleString(extractedData.quantity),
             amountUSD: safeToLocaleString(extractedData.amountUSD, 'en-US'),
             commissionUSD: safeToLocaleString(extractedData.commissionUSD, 'en-US'),
             totalUSD: safeToLocaleString(extractedData.totalUSD, 'en-US'),
             totalKRW: safeToLocaleString(extractedData.totalKRW, 'ko-KR'),
             balanceKRW: safeToLocaleString(extractedData.balanceKRW, 'ko-KR'),
        };
        
        console.log(`Formatted data for row ${index + 1}:`, formattedData);
        addRow(formattedData);
    });

    // End logging with estimated costs
    const processingTime = Date.now() - startTime;
    
    // Estimate tokens and costs
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    
    allExtractedData.forEach(data => {
        try {
            // Estimate input tokens (prompt + image)
            const promptText = "제공된 수입 정산서 문서에서 데이터를 추출해 주세요...";
            const inputTokens = estimateTokens(promptText, true); // true for image
            
            // Estimate output tokens (JSON response) - safely handle invalid data
            const outputText = data ? JSON.stringify(data) : '{}';
            const outputTokens = estimateTokens(outputText);
            
            totalInputTokens += inputTokens || 0;
            totalOutputTokens += outputTokens || 0;
        } catch (error) {
            console.warn('Error estimating tokens for data:', data, error);
        }
    });
    
    endLogging(logId, processingTime, totalInputTokens, totalOutputTokens);
    
    // Show rating section if processing was successful
    if (successCount > 0) {
        showRatingSection();
    }

    if (successCount < totalToProcess) {
        alert(`${successCount} / ${totalToProcess} 개의 페이지만 성공적으로 처리되었습니다.`);
    }
    
    } catch (globalError) {
        console.error('Critical error in processDocument:', globalError);
        alert(`처리 중 심각한 오류가 발생했습니다: ${globalError instanceof Error ? globalError.message : String(globalError)}`);
    } finally {
        // Always restore UI state
        loadingOverlay.classList.add('hidden');
        processButton.disabled = false;
        updateProcessButtonState();
        loadingOverlay.querySelector('p')!.textContent = 'AI가 문서를 분석 중입니다...';
    }
}

// Event Listeners
function setupEventListeners() {
    // Initialize elements
    aiProviderPills = document.querySelectorAll('.ai-pill') as NodeListOf<HTMLButtonElement>;
    modelSelector = document.getElementById('model-selector') as HTMLSelectElement;
    debugStatusButton = document.getElementById('debug-status-button') as HTMLButtonElement;
    
    // Update elements
    updateNotification = document.getElementById('update-notification') as HTMLDivElement;
    updateVersionSpan = document.getElementById('update-version') as HTMLSpanElement;
    updateButton = document.getElementById('update-button') as HTMLButtonElement;
    dismissUpdateButton = document.getElementById('dismiss-update') as HTMLButtonElement;
    checkUpdateButton = document.getElementById('check-update-button') as HTMLButtonElement;
    
    console.log('🔧 [Debug] Debug button element:', debugStatusButton);
    
    // Windows-specific debugging
    console.log('🔧 [Debug] Platform:', navigator.platform);
    console.log('🔧 [Debug] User Agent:', navigator.userAgent);
    console.log('🔧 [Debug] Language:', navigator.language);
    console.log('🔧 [Debug] Online:', navigator.onLine);
    
    // Check for potential blocking extensions
    if (typeof (window as any).chrome !== 'undefined') {
        console.log('🔧 [Debug] Chrome extension environment detected');
    }
    if (typeof (window as any).InstallTrigger !== 'undefined') {
        console.log('🔧 [Debug] Firefox environment detected');
    }
    
    // API Settings Modal elements
    apiSettingsModal = document.getElementById('api-settings-modal') as HTMLDivElement;
    apiSettingsButton = document.getElementById('api-settings-button') as HTMLButtonElement;
    closeModalButton = document.getElementById('close-modal') as HTMLButtonElement;
    modalOverlay = document.getElementById('modal-overlay') as HTMLDivElement;
    saveApiKeysButton = document.getElementById('save-api-keys') as HTMLButtonElement;
    cancelApiSettingsButton = document.getElementById('cancel-api-settings') as HTMLButtonElement;
    geminiKeyInput = document.getElementById('gemini-key') as HTMLInputElement;
    openaiKeyInput = document.getElementById('openai-key') as HTMLInputElement;
    upstageKeyInput = document.getElementById('upstage-key') as HTMLInputElement;
    // File handling
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
            handleFilesSelect(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener('change', (e) => {
        const target = e.target as HTMLInputElement;
        if (target.files && target.files.length > 0) {
            handleFilesSelect(target.files);
        }
    });

    // Processing and table
    processButton.addEventListener('click', processDocument);
    addRowButton.addEventListener('click', () => addRow());
    copyButton.addEventListener('click', copyTableToClipboard);

    tableBody.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        if (target.closest('.delete-button')) {
            const row = target.closest('tr');
            if (row?.dataset.id) {
                deleteRow(parseInt(row.dataset.id, 10));
            }
        }
    });

    tableBody.addEventListener('change', (e) => {
        const target = e.target as HTMLInputElement;
        if (target.tagName === 'INPUT') {
            const row = target.closest('tr');
            const field = target.dataset.field as keyof Omit<TableRowData, 'id'>;
            if (row?.dataset.id && field) {
                updateCell(parseInt(row.dataset.id, 10), field, target.value);
                // Update tooltip with new value
                target.title = target.value;
            }
        }
    });

    // Provider pill selection
    aiProviderPills.forEach(pill => {
        pill.addEventListener('click', async () => {
            const provider = pill.dataset.provider as 'gemini' | 'openai' | 'upstage';
            
            // Check if the selected provider is available
            if (!isProviderAvailableSync(provider)) {
                alert(`${provider.toUpperCase()} 서비스에 연결할 수 없습니다. 설정을 확인해주세요.`);
                return;
            }
            
            // Update settings
            currentSettings.provider = provider;
            currentSettings.model = getDefaultModelForProvider(provider);
            
            
            saveSettings(currentSettings);
            
            // Update UI
            updateModelSelector();
            await updateProviderPillsStatus();
            updateProcessButtonState();
        });
    });
    
    // Model selector change handler
    modelSelector.addEventListener('change', () => {
        currentSettings.model = modelSelector.value;
        saveSettings(currentSettings);
    });
    
    // Rating button handlers
    likeButton.addEventListener('click', () => {
        rateLastResult('like');
    });
    
    dislikeButton.addEventListener('click', () => {
        rateLastResult('dislike');
    });
    
    // Analytics dashboard toggle
    showAnalyticsButton.addEventListener('click', toggleAnalyticsDashboard);
    
    // Export logs button
    exportLogsButton.addEventListener('click', exportUsageLogs);
    
    // Clear logs button
    clearLogsButton.addEventListener('click', clearAllLogs);
    
    debugStatusButton.addEventListener('click', debugAPIKeyStatus);
    
    // API Settings Modal handlers
    apiSettingsButton.addEventListener('click', showAPISettingsModal);
    closeModalButton.addEventListener('click', hideAPISettingsModal);
    modalOverlay.addEventListener('click', hideAPISettingsModal);
    saveApiKeysButton.addEventListener('click', saveAPIKeysFromModal);
    cancelApiSettingsButton.addEventListener('click', hideAPISettingsModal);
    
    
    // Password visibility toggles
    document.querySelectorAll('.toggle-visibility').forEach(button => {
        button.addEventListener('click', () => {
            const target = button.getAttribute('data-target');
            if (target) {
                togglePasswordVisibility(target);
            }
        });
    });
    
    // Global keyboard shortcuts - use capture phase to ensure it's not blocked
    document.addEventListener('keydown', (e) => {
        // ESC key to close modal
        if (e.key === 'Escape' && !apiSettingsModal.classList.contains('hidden')) {
            hideAPISettingsModal();
        }
        
        // Ctrl+Z for undo (Windows/Linux) or Cmd+Z (Mac)
        if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'z' || e.code === 'KeyZ')) {
            console.log('Undo key combination detected, deleteHistory.length:', deleteHistory.length);
            if (deleteHistory.length > 0) {
                e.preventDefault();
                e.stopPropagation();
                undoLastDelete();
            }
        }
    }, true); // Use capture phase
    
    // Tab switching functionality
    tabButtons = document.querySelectorAll('.tab-button') as NodeListOf<HTMLButtonElement>;
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            if (!tabName) return;
            
            // Update active tab button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show/hide tab content
            const recentTab = document.getElementById('recent-tab');
            const statsTab = document.getElementById('stats-tab');
            
            if (tabName === 'recent') {
                recentTab?.classList.remove('hidden');
                statsTab?.classList.add('hidden');
                renderRecentUsage(); // Refresh recent usage when tab is shown
            } else if (tabName === 'stats') {
                recentTab?.classList.add('hidden');
                statsTab?.classList.remove('hidden');
            }
        });
    });
    
    // Update notification handlers
    if (dismissUpdateButton) {
        dismissUpdateButton.addEventListener('click', hideUpdateNotification);
    }
    if (checkUpdateButton) {
        checkUpdateButton.addEventListener('click', manualUpdateCheck);
    }
    if (debugStatusButton) {
        debugStatusButton.addEventListener('click', debugAPIKeyStatus);
    }
}

// Debug status function
function debugAPIKeyStatus() {
    console.log('🔧 [Debug Status] ========== API Key Debug Info ==========');
    
    // Check localStorage
    const localStorageKeys = localStorage.getItem('secureAPIKeys');
    console.log('🔧 [Debug Status] localStorage secureAPIKeys:', localStorageKeys ? 'exists' : 'not found');
    if (localStorageKeys) {
        try {
            const parsed = JSON.parse(localStorageKeys);
            console.log('🔧 [Debug Status] localStorage keys:', Object.keys(parsed));
        } catch (e) {
            console.log('🔧 [Debug Status] localStorage parse error:', e);
        }
    }
    
    // Check cached keys
    console.log('🔧 [Debug Status] cachedAPIKeys:', cachedAPIKeys ? Object.keys(cachedAPIKeys) : 'null');
    if (cachedAPIKeys) {
        Object.entries(cachedAPIKeys).forEach(([provider, key]) => {
            console.log(`🔧 [Debug Status] ${provider}: ${key ? `${key.length} chars` : 'empty'}`);
        });
    }
    
    // Check getAPIKey for each provider
    const providers = ['gemini', 'openai', 'upstage'];
    providers.forEach(provider => {
        const key = getAPIKey(provider);
        console.log(`🔧 [Debug Status] getAPIKey(${provider}): ${key ? `${key.length} chars` : 'not found'}`);
    });
    
    // Check environment variables
    console.log('🔧 [Debug Status] Environment variables:');
    console.log('🔧 [Debug Status] VITE_GEMINI_API_KEY:', (import.meta as any).env?.VITE_GEMINI_API_KEY ? 'exists' : 'not found');
    console.log('🔧 [Debug Status] VITE_OPENAI_API_KEY:', (import.meta as any).env?.VITE_OPENAI_API_KEY ? 'exists' : 'not found');
    console.log('🔧 [Debug Status] VITE_UPSTAGE_API_KEY:', (import.meta as any).env?.VITE_UPSTAGE_API_KEY ? 'exists' : 'not found');
    
    console.log('🔧 [Debug Status] ========================================');
    
    // Force provider status update
    console.log('🔧 [Debug Status] Forcing provider status update...');
    updateProviderPillsStatus();
}

// Network connectivity test for Windows debugging
async function testNetworkConnectivity() {
    console.log('🔧 [Network Test] Starting connectivity tests...');
    
    const tests = [
        { name: 'Google DNS', url: 'https://dns.google/resolve?name=google.com&type=A' },
        { name: 'Gemini API', url: 'https://generativelanguage.googleapis.com/' },
        { name: 'OpenAI API', url: 'https://api.openai.com/' },
        { name: 'Upstage API', url: 'https://api.upstage.ai/' }
    ];
    
    for (const test of tests) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const start = Date.now();
            const response = await fetch(test.url, {
                method: 'HEAD',
                signal: controller.signal
            });
            const duration = Date.now() - start;
            
            clearTimeout(timeoutId);
            console.log(`✅ [Network Test] ${test.name}: ${response.status} (${duration}ms)`);
        } catch (error) {
            console.error(`❌ [Network Test] ${test.name} failed:`, error.message);
        }
    }
}

// Initial setup
document.addEventListener('DOMContentLoaded', async () => {
    // Load usage logs
    usageLogs = loadUsageLogs();
    
    // Initialize API keys
    try {
        cachedAPIKeys = loadAPIKeysFromStorage();
        console.log('API keys loaded successfully');
    } catch (error) {
        console.log('Failed to load API keys:', error);
    }
    
    initializeSettings();
    setupEventListeners();
    initializeUI();
    updateModelSelector();
    renderTable();
    setupInputTooltips();
    
    // Display current version
    const versionElement = document.getElementById('current-version');
    if (versionElement) {
        versionElement.textContent = `v${APP_CONFIG.version}`;
    }
    
    // Initialize update checker
    initializeUpdateChecker();
    
    // Windows-specific network connectivity test
    if (navigator.userAgent.toLowerCase().includes('win')) {
        console.log('🔧 [Debug] Windows detected, running network tests...');
        setTimeout(() => testNetworkConnectivity(), 2000); // Run after 2 seconds
    }
});