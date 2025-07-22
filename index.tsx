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
    provider: 'gemini' | 'openai' | 'upstage' | 'ollama';
    model: string;
    localEndpoint?: string; // For local models
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
    
    // Local models (free)
    'llama3.2-vision:11b': { input: 0, output: 0 },
    'llava:13b': { input: 0, output: 0 },
    'moondream:latest': { input: 0, output: 0 }
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
    ],
    ollama: [
        { id: 'llama3.2-vision:11b', name: 'Llama 3.2 Vision 11B', description: '로컬 비전 모델' },
        { id: 'llava:13b', name: 'LLaVA 13B', description: '로컬 멀티모달 모델' },
        { id: 'moondream:latest', name: 'Moondream', description: '경량 비전 모델' }
    ]
};

// Application configuration
const APP_CONFIG = {
    version: '1.0.0',
    githubRepo: 'your-username/auto-scan-app', // TODO: Replace with actual GitHub repository
    checkForUpdates: true
};

// Global state
let tableData: TableRowData[] = [];
let nextId = 0;
let selectedPages: PageData[] = [];
let pdfFileGroups: Map<string, PageData[]> = new Map(); // Group pages by filename
let currentSettings: AISettings = {
    provider: 'gemini',
    model: 'gemini-2.5-flash',
    localEndpoint: 'http://localhost:11434' // Default Ollama endpoint
};

// Usage logging
let usageLogs: UsageLog[] = [];
let currentLogId: string | null = null;

// Get local endpoint for Ollama (only provider that doesn't use proxy)
const getLocalEndpoint = (provider: string): string => {
    const env = (import.meta as any).env;
    switch (provider) {
        case 'ollama':
            return env?.VITE_OLLAMA_ENDPOINT || 'http://localhost:11434';
        default:
            return '';
    }
};

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

// Update elements
let updateNotification: HTMLDivElement;
let updateVersionSpan: HTMLSpanElement;
let updateButton: HTMLButtonElement;
let dismissUpdateButton: HTMLButtonElement;

// API Settings Modal elements
let apiSettingsModal: HTMLDivElement;
let apiSettingsButton: HTMLButtonElement;
let closeModalButton: HTMLButtonElement;
let modalOverlay: HTMLDivElement;
let saveApiKeysButton: HTMLButtonElement;
let cancelApiSettingsButton: HTMLButtonElement;
let geminiKeyInput: HTMLInputElement;
let openaiKeyInput: HTMLInputElement;
let claudeKeyInput: HTMLInputElement;
let upstageKeyInput: HTMLInputElement;

// Check if provider is available (proxy server or local endpoint)
async function isProviderAvailable(provider: string): Promise<boolean> {
    // For proxy server providers, check if proxy server is running
    if (['gemini', 'openai', 'upstage'].includes(provider)) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 1000);
            
            const response = await fetch('http://localhost:3003/health', {
                method: 'GET',
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            return response.ok;
        } catch (error) {
            return false;
        }
    }
    
    // For local models, check if endpoint is accessible
    if (provider === 'ollama') {
        try {
            const endpoint = getLocalEndpoint(provider);
            if (!endpoint) return false;
            
            const healthEndpoint = '/api/tags';
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 1000);
            
            const response = await fetch(endpoint + healthEndpoint, {
                method: 'GET',
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            return response.ok;
        } catch (error) {
            return false;
        }
    }
    
    return false;
}

// Synchronous version for backward compatibility
function isProviderAvailableSync(provider: string): boolean {
    // For proxy server providers, check if API key is available
    if (['gemini', 'openai', 'upstage', 'claude'].includes(provider)) {
        const apiKey = getAPIKey(provider);
        return !!apiKey;
    }
    
    // For local models, check if endpoint is configured
    if (provider === 'ollama') {
        const endpoint = getLocalEndpoint(provider);
        return !!endpoint;
    }
    
    return false;
}

// Update provider pill status
async function updateProviderPillsStatus() {
    if (!aiProviderPills) return;
    
    for (const pill of aiProviderPills) {
        const provider = pill.dataset.provider as string;
        const statusIndicator = pill.querySelector('.pill-status') as HTMLElement;
        const isAvailable = await isProviderAvailable(provider);
        
        // Remove existing classes
        pill.classList.remove('active', 'available', 'unavailable');
        statusIndicator.classList.remove('available', 'unavailable');
        
        // Update availability status
        if (isAvailable) {
            pill.classList.add('available');
            statusIndicator.classList.add('available');
        } else {
            pill.classList.add('unavailable');
            statusIndicator.classList.add('unavailable');
        }
        
        // Mark current selection
        if (provider === currentSettings.provider) {
            pill.classList.add('active');
        }
    }
    
    // Update model selector
    updateModelSelector();
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

// Update available models list for local providers
async function updateAvailableModelsForLocalProvider(provider: 'ollama') {
    try {
        const endpoint = getLocalEndpoint(provider);
        let models: Array<{id: string, name: string, description: string}> = [];
        
        const response = await fetch(endpoint + '/api/tags');
        const data = await response.json();
        models = data.models?.map((model: any) => ({
            id: model.name,
            name: model.name,
            description: model.details?.parameter_size || 'Ollama 모델'
        })) || [];
        
        // Update the PROVIDER_MODELS for this provider
        PROVIDER_MODELS[provider] = models;
        
        // Update UI if this is the current provider
        if (currentSettings.provider === provider) {
            updateModelSelector();
        }
    } catch (error) {
        console.warn(`Failed to fetch models for ${provider}:`, error);
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
interface GitHubRelease {
    tag_name: string;
    name: string;
    html_url: string;
    published_at: string;
    body: string;
}

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

async function checkForUpdates(): Promise<GitHubRelease | null> {
    if (!APP_CONFIG.checkForUpdates) return null;
    
    try {
        const response = await fetch(`https://api.github.com/repos/${APP_CONFIG.githubRepo}/releases/latest`, {
            headers: {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Auto-Scan-App'
            }
        });
        
        if (!response.ok) {
            if (response.status === 404) {
                console.log('No releases found for this repository');
                return null;
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const release: GitHubRelease = await response.json();
        const currentVersion = APP_CONFIG.version;
        const latestVersion = release.tag_name;
        
        if (compareVersions(currentVersion, latestVersion) < 0) {
            return release;
        }
        
        return null;
    } catch (error) {
        console.error('Failed to check for updates:', error);
        return null;
    }
}

function showUpdateNotification(release: GitHubRelease) {
    if (!updateNotification || !updateVersionSpan) return;
    
    updateVersionSpan.textContent = `버전 ${release.tag_name}이 사용 가능합니다.`;
    updateNotification.classList.remove('hidden');
    
    // Store release info for later use
    updateButton.onclick = () => {
        window.open(release.html_url, '_blank');
        hideUpdateNotification();
    };
}

function hideUpdateNotification() {
    if (!updateNotification) return;
    updateNotification.classList.add('hidden');
    
    // Remember that user dismissed this version
    localStorage.setItem('dismissedUpdate', APP_CONFIG.version);
}

async function initializeUpdateChecker() {
    if (!APP_CONFIG.checkForUpdates) return;
    
    // Don't check too frequently - limit to once per day
    const lastCheckTime = localStorage.getItem('lastUpdateCheck');
    const now = Date.now();
    const dayInMs = 24 * 60 * 60 * 1000;
    
    if (lastCheckTime && (now - parseInt(lastCheckTime)) < dayInMs) {
        return;
    }
    
    const dismissedVersion = localStorage.getItem('dismissedUpdate');
    if (dismissedVersion === APP_CONFIG.version) {
        return;
    }
    
    try {
        const release = await checkForUpdates();
        if (release) {
            // Wait a bit before showing notification
            setTimeout(() => showUpdateNotification(release), 2000);
        }
        localStorage.setItem('lastUpdateCheck', now.toString());
    } catch (error) {
        console.error('Update check failed:', error);
    }
}

// API Key Security Functions
function generateKey(): string {
    // Generate a simple key for basic encryption (client-side)
    const array = new Uint8Array(16);
    crypto.getRandomValues(array);
    return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
}

function getOrCreateEncryptionKey(): string {
    let key = localStorage.getItem('encryptionKey');
    if (!key) {
        key = generateKey();
        localStorage.setItem('encryptionKey', key);
    }
    return key;
}

function simpleEncrypt(text: string, key: string): string {
    if (!text) return '';
    let encrypted = '';
    for (let i = 0; i < text.length; i++) {
        const charCode = text.charCodeAt(i) ^ key.charCodeAt(i % key.length);
        encrypted += String.fromCharCode(charCode);
    }
    return btoa(encrypted);
}

function simpleDecrypt(encryptedText: string, key: string): string {
    if (!encryptedText) return '';
    try {
        const text = atob(encryptedText);
        let decrypted = '';
        for (let i = 0; i < text.length; i++) {
            const charCode = text.charCodeAt(i) ^ key.charCodeAt(i % key.length);
            decrypted += String.fromCharCode(charCode);
        }
        return decrypted;
    } catch (error) {
        console.error('Failed to decrypt:', error);
        return '';
    }
}

interface SecureAPIKeys {
    gemini?: string;
    openai?: string;
    claude?: string;
    upstage?: string;
}

function saveSecureAPIKeys(keys: SecureAPIKeys) {
    const encryptionKey = getOrCreateEncryptionKey();
    const encryptedKeys: any = {};
    
    Object.entries(keys).forEach(([provider, key]) => {
        if (key && key.trim()) {
            encryptedKeys[provider] = simpleEncrypt(key.trim(), encryptionKey);
        }
    });
    
    localStorage.setItem('secureAPIKeys', JSON.stringify(encryptedKeys));
}

function loadSecureAPIKeys(): SecureAPIKeys {
    const stored = localStorage.getItem('secureAPIKeys');
    if (!stored) return {};
    
    try {
        const encryptedKeys = JSON.parse(stored);
        const encryptionKey = getOrCreateEncryptionKey();
        const decryptedKeys: SecureAPIKeys = {};
        
        Object.entries(encryptedKeys).forEach(([provider, encryptedKey]) => {
            if (typeof encryptedKey === 'string') {
                const decrypted = simpleDecrypt(encryptedKey, encryptionKey);
                if (decrypted) {
                    decryptedKeys[provider as keyof SecureAPIKeys] = decrypted;
                }
            }
        });
        
        return decryptedKeys;
    } catch (error) {
        console.error('Failed to load API keys:', error);
        return {};
    }
}

function getAPIKey(provider: string): string {
    // First try to get from environment variables (for development)
    const envKey = (import.meta as any).env?.[`VITE_${provider.toUpperCase()}_API_KEY`];
    if (envKey) return envKey;
    
    // Then try to get from secure storage
    const keys = loadSecureAPIKeys();
    const key = keys[provider as keyof SecureAPIKeys];
    return key || '';
}

// Modal Management Functions
function showAPISettingsModal() {
    if (!apiSettingsModal) return;
    
    // Load existing keys
    const keys = loadSecureAPIKeys();
    if (geminiKeyInput) geminiKeyInput.value = keys.gemini || '';
    if (openaiKeyInput) openaiKeyInput.value = keys.openai || '';
    if (claudeKeyInput) claudeKeyInput.value = keys.claude || '';
    if (upstageKeyInput) upstageKeyInput.value = keys.upstage || '';
    
    apiSettingsModal.classList.remove('hidden');
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
    if (claudeKeyInput) claudeKeyInput.value = '';
    if (upstageKeyInput) upstageKeyInput.value = '';
}

function saveAPIKeysFromModal() {
    const keys: SecureAPIKeys = {};
    
    if (geminiKeyInput?.value.trim()) keys.gemini = geminiKeyInput.value.trim();
    if (openaiKeyInput?.value.trim()) keys.openai = openaiKeyInput.value.trim();
    if (claudeKeyInput?.value.trim()) keys.claude = claudeKeyInput.value.trim();
    if (upstageKeyInput?.value.trim()) keys.upstage = upstageKeyInput.value.trim();
    
    saveSecureAPIKeys(keys);
    hideAPISettingsModal();
    
    // Refresh provider status
    updateProviderPillsStatus();
    
    // Show success message
    alert('API 키가 안전하게 저장되었습니다.');
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
    updateProviderPillsStatus();
    updateProcessButtonState();
}

// Update process button state based on settings and file selection
function updateProcessButtonState() {
    const hasApiKey = isProviderAvailableSync(currentSettings.provider);
    const hasSelectedPages = selectedPages.length > 0;
    processButton.disabled = !hasApiKey || !hasSelectedPages;
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
    
    // Update UI
    renderPdfPages();
    
    // Update preview if needed
    if (selectedPages.length === 0) {
        imagePreview.classList.add('hidden');
        dropZone.querySelector('p')?.classList.remove('hidden');
    }
    
    // Hide PDF preview container if no files
    if (pdfFileGroups.size === 0) {
        pdfPreviewContainer.classList.add('hidden');
    }
    
    updateProcessButtonState();
}

function handleImageFile(file: File) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const result = e.target?.result as string;
        const pageData: PageData = {
            data: result.split(',')[1],
            mimeType: file.type,
            fileName: file.name
        };
        selectedPages.push(pageData);
        imagePreview.src = result;
        imagePreview.classList.remove('hidden');
        dropZone.querySelector('p')?.classList.add('hidden');
        updateProcessButtonState();
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
            pdfPreviewContainer.classList.remove('hidden');
            
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
                const pageInfo: PageData = {
                    data: pageDataUrl.split(',')[1],
                    mimeType: 'image/png',
                    fileName: file.name,
                    pageNumber: i
                };
                pages.push(pageInfo);
            }
            
            // Add pages to file groups
            pdfFileGroups.set(file.name, pages);
            
            // Render the updated PDF pages
            renderPdfPages();
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
    if (selectedPages.length === 0) {
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

// AI Processing Functions
async function processWithGemini(pageData: PageData) {
    const textPart = {
        text: "제공된 수입 정산서 문서에서 정확한 항목별로 데이터를 추출해 주세요:\n\n1. date: 문서의 작성일 (YYYY-MM-DD 형식)\n2. quantity: 수량 (GT 단위)\n3. amountUSD: COMMERCIAL INVOICE CHARGE의 US$ 금액\n4. commissionUSD: COMMISSION의 US$ 금액\n5. totalUSD: '입금하신 금액' 또는 '수수료포함금액'의 US$ 금액 (총 경비가 아님)\n6. totalKRW: '입금하신 금액' 또는 '수수료포함금액'의 원화(₩) 금액 (총 경비가 아님)\n7. balanceKRW: 잔액의 원화(₩) 금액\n\n주의사항: totalUSD와 totalKRW는 반드시 '입금하신 금액' 섹션에서 추출하세요."
    };

    const imagePart = {
        inlineData: { mimeType: pageData.mimeType, data: pageData.data },
    };

    const response = await fetch('http://localhost:3003/api/gemini', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: currentSettings.model,
            contents: { parts: [textPart, imagePart] },
            config: {
                responseMimeType: "application/json",
            }
        })
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Gemini API 오류: ${response.status} - ${errorData.details || errorData.error}`);
    }

    const result = await response.json();
    const jsonText = result.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || '';
    if (!jsonText) {
        throw new Error('AI 응답이 비어있습니다.');
    }
    return JSON.parse(jsonText);
}

async function processWithOpenAI(pageData: PageData) {
    // Always use proxy server for security
    const response = await fetch('http://localhost:3003/api/openai', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
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
    return JSON.parse(content);
}


async function processWithUpstage(pageData: PageData) {
    // Use proxy server for security
    const isDocVision = currentSettings.model === 'solar-docvision-preview';
    
    if (isDocVision) {
        // --- 추가된 코드 시작 ---
        const requestBody = {
            model: currentSettings.model,
            messages: [
                {
                    role: "user",
                    content: [
                        { type: "text", text: "제공된 수입 정산서 문서에서 정확한 항목별로 데이터를 추출해 주세요:\n\n1. date: 문서의 작성일 (YYYY-MM-DD 형식)\n2. quantity: 수량 (GT 단위)\n3. amountUSD: COMMERCIAL INVOICE CHARGE의 US$ 금액\n4. commissionUSD: COMMISSION의 US$ 금액\n5. totalUSD: '입금하신 금액' 또는 '수수료포함금액'의 US$ 금액 (총 경비가 아님)\n6. totalKRW: '입금하신 금액' 또는 '수수료포함금액'의 원화(₩) 금액 (총 경비가 아님)\n7. balanceKRW: 잔액의 원화(₩) 금액\n\n주의사항: totalUSD와 totalKRW는 반드시 '입금하신 금액' 섹션에서 추출하세요.\n\nJSON 형식으로 반환: {\"date\": \"YYYY-MM-DD\", \"quantity\": 숫자, \"amountUSD\": 숫자, \"commissionUSD\": 숫자, \"totalUSD\": 숫자, \"totalKRW\": 숫자, \"balanceKRW\": 숫자}" },
                        { type: "image_url", image_url: { url: `data:${pageData.mimeType};base64,${pageData.data}` } }
                    ]
                }
            ],
            stream: false
        };
        await logToFile(JSON.stringify(requestBody, null, 2), 'upstage-docvision-input.json');
        // --- 추가된 코드 끝 ---
        // Solar DocVision uses chat completions format
        const response = await fetch(apiEndpoint, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorText = await response.text();
            // --- 추가된 코드 시작 ---
            await logToFile(errorText, 'upstage-docvision-error.txt');
            // --- 추가된 코드 끝 ---
            throw new Error(`Upstage DocVision API 오류: ${response.status} ${response.statusText} - ${errorText}`);
        }

        const result = await response.json();
        // --- 추가된 코드 시작 ---
        await logToFile(JSON.stringify(result, null, 2), 'upstage-docvision-output.json');
        // --- 추가된 코드 끝 ---
        
        // Chat API 응답에서 JSON 데이터 추출
        try {
            const content = result.choices?.[0]?.message?.content;
            if (!content) {
                throw new Error('Upstage DocVision API에서 응답을 받지 못했습니다.');
            }
            
            const jsonMatch = content.match(/\{[\s\S]*?\}/);
            if (jsonMatch) {
                return JSON.parse(jsonMatch[0]);
            } else {
                console.log('No JSON found in DocVision response:', content);
                throw new Error('JSON 형식을 찾을 수 없습니다. 응답: ' + content.substring(0, 100));
            }
        } catch (parseError) {
            console.error('Upstage DocVision 응답 파싱 오류:', parseError);
            throw new Error('Upstage DocVision 응답을 파싱할 수 없습니다.');
        }
    } else {
        // Document Parse API - try multipart/form-data format
        const formData = new FormData();
        
        // Convert base64 to blob
        const byteCharacters = atob(pageData.data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: pageData.mimeType });
        
        formData.append('model', 'document-parse');  // Fixed model name
        formData.append('document', blob, 'document.' + (pageData.mimeType.includes('png') ? 'png' : 'jpg'));
        formData.append('ocr', 'auto');  // Required field
        formData.append('output_formats', JSON.stringify(['text']));  // Proper array format
        
        // --- 추가된 코드 시작 ---
        const formDataLog = `
--- FormData Fields ---
model: document-parse
document: [Blob, type=${blob.type}, size=${blob.size}]
ocr: auto
output_formats: ["text"]
--- Image Data (first 100 chars of base64) ---
${pageData.data.substring(0, 100)}...
`;
        await logToFile(formDataLog, 'upstage-parse-input.txt');
        // --- 추가된 코드 끝 ---
        const response = await fetch(apiEndpoint, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                // Don't set Content-Type for FormData, let browser set it with boundary
            },
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            // --- 추가된 코드 시작 ---
            await logToFile(errorText, 'upstage-parse-error.txt');
            // --- 추가된 코드 끝 ---
            throw new Error(`Upstage Document Parse API 오류: ${response.status} ${response.statusText} - ${errorText}`);
        }

        const result = await response.json();
        // --- 추가된 코드 시작 ---
        await logToFile(JSON.stringify(result, null, 2), 'upstage-parse-output.json');
        // --- 추가된 코드 끝 ---
        
        // Document Parse API 응답에서 elements를 활용한 키워드 기반 데이터 추출
        try {
            console.log('Full Upstage Document Parse response:', result);
            
            const extractedData = {
                date: '',
                quantity: 0,
                amountUSD: 0,
                commissionUSD: 0,
                totalUSD: 0,
                totalKRW: 0,
                balanceKRW: 0
            };

            // elements 배열에서 구조화된 정보 추출
            if (result.elements && result.elements.length > 0) {
                console.log('Processing elements:', result.elements.length);
                
                // 날짜 추출: "일 자" 또는 "작성일"이 포함된 element에서 찾기
                const dateElement = result.elements.find((el: any) => 
                    el.content?.text && (
                        el.content.text.includes('일 자') || 
                        el.content.text.includes('작성일') ||
                        el.content.text.match(/\d{4}년\s*\d{1,2}월\s*\d{1,2}일/) ||
                        el.content.text.match(/\d{4}\.\d{2}\.\d{2}/)
                    )
                );
                
                if (dateElement) {
                    const dateText = dateElement.content.text;
                    const datePatterns = [
                        /(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)/,
                        /(\d{4}\.\d{2}\.\d{2})/,
                        /(\d{4}-\d{2}-\d{2})/
                    ];
                    
                    for (const pattern of datePatterns) {
                        const match = dateText.match(pattern);
                        if (match) {
                            let dateStr = match[1];
                            if (dateStr.includes('년')) {
                                dateStr = dateStr.replace(/년\s*/g, '-').replace(/월\s*/g, '-').replace(/일/g, '');
                                // 월과 일이 한 자리수인 경우 앞에 0 추가
                                const parts = dateStr.split('-');
                                if (parts.length === 3) {
                                    dateStr = `${parts[0]}-${parts[1].padStart(2, '0')}-${parts[2].padStart(2, '0')}`;
                                }
                            } else {
                                dateStr = dateStr.replace(/\./g, '-');
                            }
                            extractedData.date = dateStr;
                            console.log('Found date:', dateStr);
                            break;
                        }
                    }
                }

                // 수량 추출: "GT" 키워드가 있는 element에서 찾기
                const quantityElement = result.elements.find((el: any) => 
                    el.content?.text && el.content.text.includes('GT')
                );
                
                if (quantityElement) {
                    const quantityMatch = quantityElement.content.text.match(/수\s*량\s*([\d,]+)\s*GT/i) ||
                                        quantityElement.content.text.match(/([\d,]+)\s*GT/i);
                    if (quantityMatch) {
                        extractedData.quantity = parseFloat(quantityMatch[1].replace(/,/g, ''));
                        console.log('Found quantity:', extractedData.quantity);
                    }
                }

                // 금액 관련 데이터가 있는 table element 찾기 (제품비용 섹션)
                const amountElement = result.elements.find((el: any) => 
                    el.content?.text && el.category === 'table' && (
                        el.content.text.includes('COMMERCIAL INVOICE') ||
                        el.content.text.includes('COMMISSION') ||
                        el.content.text.includes('제품비용')
                    )
                );

                if (amountElement) {
                    const amountText = amountElement.content.text;
                    console.log('Processing amount text:', amountText);
                    
                    // 통화 기호 패턴 (OCR로 인한 변형 고려: ₩, \, 원 등)
                    const currencyPattern = '[₩\\\\원]?';
                    
                    // COMMERCIAL INVOICE CHARGE 금액 추출
                    // 실제 형식: "COMMERCIAL INVOICE CARGE ₩32,744,630 ₩3,274,463 US$22,234.42"
                    const invoiceMatch = amountText.match(
                        new RegExp(`COMMERCIAL\\s+INVOICE\\s+CAR?GE?\\s+${currencyPattern}([\\d,]+)\\s+${currencyPattern}[\\d,]+\\s+US\\$([\\d,]+(?:\\.\\d+)?)`, 'i')
                    );
                    if (invoiceMatch) {
                        extractedData.amountUSD = parseFloat(invoiceMatch[2].replace(/,/g, ''));
                        console.log('Found amountUSD:', extractedData.amountUSD);
                    }
                    
                    // COMMISSION 금액 추출
                    // 실제 형식: "COMMISSION ₩327,440 ₩32,744 US$222.34"
                    const commissionMatch = amountText.match(
                        new RegExp(`COMMISSION\\s+${currencyPattern}([\\d,]+)\\s+${currencyPattern}[\\d,]+\\s+US\\$([\\d,]+(?:\\.\\d+)?)`, 'i')
                    );
                    if (commissionMatch) {
                        extractedData.commissionUSD = parseFloat(commissionMatch[2].replace(/,/g, ''));
                        console.log('Found commissionUSD:', extractedData.commissionUSD);
                    }
                    
                    // TOTAL 2번 (제품비용 합계) 추출
                    // 실제 형식: "TOTAL 2번 ₩33,072,070 ₩3,307,207 US$22,456.76"
                    const total2Match = amountText.match(
                        new RegExp(`TOTAL\\s+2번\\s+${currencyPattern}([\\d,]+)\\s+${currencyPattern}[\\d,]+\\s+US\\$([\\d,]+(?:\\.\\d+)?)`, 'i')
                    );
                    if (total2Match) {
                        extractedData.totalUSD = parseFloat(total2Match[2].replace(/,/g, ''));
                        extractedData.totalKRW = parseFloat(total2Match[1].replace(/,/g, ''));
                        console.log('Found totalUSD:', extractedData.totalUSD);
                        console.log('Found totalKRW:', extractedData.totalKRW);
                    }
                }

                // 잔액이 있는 element 찾기 (하단 정산 섹션)
                const balanceElement = result.elements.find((el: any) => 
                    el.content?.text && (
                        el.content.text.includes('잔 액') ||
                        el.content.text.includes('잔액')
                    )
                );

                if (balanceElement) {
                    const balanceText = balanceElement.content.text;
                    console.log('Processing balance text:', balanceText);
                    
                    // OCR로 인한 통화 기호 변형 고려 (₩, \, 원, 없음 등)
                    // 실제 형식: "잔 액 \4,796,651" (₩가 \로 잘못 인식됨)
                    const balanceMatch = balanceText.match(/잔\s*액\s*[₩\\원]?([\d,]+)/i);
                    if (balanceMatch) {
                        extractedData.balanceKRW = parseFloat(balanceMatch[1].replace(/,/g, ''));
                        console.log('Found balanceKRW:', extractedData.balanceKRW);
                    }
                }
                
            } else {
                // Fallback: content.text가 있는 경우 기존 방식 사용
                let extractedText = '';
                
                if (result.content && result.content.text) {
                    extractedText = result.content.text;
                } else if (result.content && result.content.markdown) {
                    extractedText = result.content.markdown;
                } else {
                    throw new Error('예상하지 못한 응답 구조입니다.');
                }
                
                if (!extractedText) {
                    throw new Error('추출된 텍스트가 비어있습니다.');
                }
                
                console.log('Using fallback text extraction method');
                
                // 기존 fallback 로직 (간소화된 버전)
                const dateMatch = extractedText.match(/(\d{4}년\s*\d{1,2}월\s*\d{1,2}일|\d{4}\.\d{2}\.\d{2})/);
                if (dateMatch) {
                    let dateStr = dateMatch[1];
                    if (dateStr.includes('년')) {
                        dateStr = dateStr.replace(/년\s*/g, '-').replace(/월\s*/g, '-').replace(/일/g, '');
                    } else {
                        dateStr = dateStr.replace(/\./g, '-');
                    }
                    extractedData.date = dateStr;
                }
                
                const quantityMatch = extractedText.match(/([\d,]+)\s*GT/i);
                if (quantityMatch) {
                    extractedData.quantity = parseFloat(quantityMatch[1].replace(/,/g, ''));
                }
            }
            
            console.log('Final extracted structured data:', extractedData);
            return extractedData;
            
        } catch (parseError) {
            console.error('Upstage Document Parse 응답 파싱 오류:', parseError);
            console.error('Original response:', result);
            throw new Error(`Upstage Document Parse 응답을 파싱할 수 없습니다: ${parseError instanceof Error ? parseError.message : String(parseError)}`);
        }
    }
}

// Local AI Processing Functions
async function processWithOllama(pageData: PageData) {
    const endpoint = getLocalEndpoint('ollama');
    
    const response = await fetch(endpoint + '/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: currentSettings.model,
            prompt: "제공된 수입 정산서 문서에서 정확한 항목별로 데이터를 추출해 주세요:\n\n1. date: 문서의 작성일 (YYYY-MM-DD 형식)\n2. quantity: 수량 (GT 단위)\n3. amountUSD: COMMERCIAL INVOICE CHARGE의 US$ 금액\n4. commissionUSD: COMMISSION의 US$ 금액\n5. totalUSD: '입금하신 금액' 또는 '수수료포함금액'의 US$ 금액 (총 경비가 아님)\n6. totalKRW: '입금하신 금액' 또는 '수수료포함금액'의 원화(₩) 금액 (총 경비가 아님)\n7. balanceKRW: 잔액의 원화(₩) 금액\n\n주의사항: totalUSD와 totalKRW는 반드시 '입금하신 금액' 섹션에서 추출하세요.\n\nJSON 형식으로 반환: {\"date\": \"YYYY-MM-DD\", \"quantity\": 숫자, \"amountUSD\": 숫자, \"commissionUSD\": 숫자, \"totalUSD\": 숫자, \"totalKRW\": 숫자, \"balanceKRW\": 숫자}",
            images: [pageData.data],
            stream: false,
            options: {
                temperature: 0.1,
                top_p: 0.9
            }
        })
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Ollama API 오류: ${response.status} - ${errorText}`);
    }

    const result = await response.json();
    
    try {
        const content = result.response;
        if (!content) {
            throw new Error('Ollama에서 응답을 받지 못했습니다.');
        }
        
        // Extract JSON from response
        const jsonMatch = content.match(/\{[\s\S]*?\}/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[0]);
        } else {
            throw new Error('JSON 형식을 찾을 수 없습니다.');
        }
    } catch (parseError) {
        console.error('Ollama 응답 파싱 오류:', parseError);
        throw new Error('Ollama 응답을 파싱할 수 없습니다.');
    }
}


async function processDocument() {
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
    
    // Start logging
    const startTime = Date.now();
    const logId = startLogging(currentSettings.provider, currentSettings.model, totalToProcess);

    for (let i = 0; i < totalToProcess; i++) {
        const page = selectedPages[i];
        loadingOverlay.querySelector('p')!.textContent = `AI가 문서를 분석 중입니다... (${i + 1}/${totalToProcess})`;

        try {
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
                case 'ollama':
                    extractedData = await processWithOllama(page);
                    break;
                default:
                    throw new Error('지원되지 않는 AI 제공자입니다.');
            }

            allExtractedData.push(extractedData);
            successCount++;

        } catch (error) {
            console.error(`Error processing page ${i + 1}:`, error);
            alert(`페이지 ${i + 1} 처리 중 오류가 발생했습니다: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
    
    allExtractedData.forEach(extractedData => {
        const formattedData = {
             date: extractedData.date,
             quantity: extractedData.quantity.toLocaleString(),
             amountUSD: extractedData.amountUSD.toLocaleString('en-US'),
             commissionUSD: extractedData.commissionUSD.toLocaleString('en-US'),
             totalUSD: extractedData.totalUSD.toLocaleString('en-US'),
             totalKRW: extractedData.totalKRW.toLocaleString('ko-KR'),
             balanceKRW: extractedData.balanceKRW.toLocaleString('ko-KR'),
        };
        addRow(formattedData);
    });

    // End logging with estimated costs
    const processingTime = Date.now() - startTime;
    
    // Estimate tokens and costs
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    
    allExtractedData.forEach(data => {
        // Estimate input tokens (prompt + image)
        const promptText = "제공된 수입 정산서 문서에서 데이터를 추출해 주세요...";
        const inputTokens = estimateTokens(promptText, true); // true for image
        
        // Estimate output tokens (JSON response)
        const outputText = JSON.stringify(data);
        const outputTokens = estimateTokens(outputText);
        
        totalInputTokens += inputTokens;
        totalOutputTokens += outputTokens;
    });
    
    endLogging(logId, processingTime, totalInputTokens, totalOutputTokens);
    
    // Show rating section if processing was successful
    if (successCount > 0) {
        showRatingSection();
    }

    loadingOverlay.classList.add('hidden');
    updateProcessButtonState();
    loadingOverlay.querySelector('p')!.textContent = 'AI가 문서를 분석 중입니다...';

    if (successCount < totalToProcess) {
        alert(`${successCount} / ${totalToProcess} 개의 페이지만 성공적으로 처리되었습니다.`);
    }
}

// Event Listeners
function setupEventListeners() {
    // Initialize elements
    aiProviderPills = document.querySelectorAll('.ai-pill') as NodeListOf<HTMLButtonElement>;
    modelSelector = document.getElementById('model-selector') as HTMLSelectElement;
    updateNotification = document.getElementById('update-notification') as HTMLDivElement;
    updateVersionSpan = document.getElementById('update-version') as HTMLSpanElement;
    updateButton = document.getElementById('update-button') as HTMLButtonElement;
    dismissUpdateButton = document.getElementById('dismiss-update') as HTMLButtonElement;
    
    // API Settings Modal elements
    apiSettingsModal = document.getElementById('api-settings-modal') as HTMLDivElement;
    apiSettingsButton = document.getElementById('api-settings-button') as HTMLButtonElement;
    closeModalButton = document.getElementById('close-modal') as HTMLButtonElement;
    modalOverlay = document.getElementById('modal-overlay') as HTMLDivElement;
    saveApiKeysButton = document.getElementById('save-api-keys') as HTMLButtonElement;
    cancelApiSettingsButton = document.getElementById('cancel-api-settings') as HTMLButtonElement;
    geminiKeyInput = document.getElementById('gemini-key') as HTMLInputElement;
    openaiKeyInput = document.getElementById('openai-key') as HTMLInputElement;
    claudeKeyInput = document.getElementById('claude-key') as HTMLInputElement;
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
            const provider = pill.dataset.provider as 'gemini' | 'openai' | 'upstage' | 'ollama';
            
            // Check if the selected provider is available
            if (!isProviderAvailableSync(provider)) {
                alert(`${provider.toUpperCase()} 서비스에 연결할 수 없습니다. 설정을 확인해주세요.`);
                return;
            }
            
            // Update settings
            currentSettings.provider = provider;
            currentSettings.model = getDefaultModelForProvider(provider);
            
            // For local providers, fetch available models
            if (provider === 'ollama') {
                await updateAvailableModelsForLocalProvider('ollama');
                const availableModels = getAvailableModels();
                if (availableModels.length > 0) {
                    currentSettings.model = availableModels[0].id;
                }
            }
            
            saveSettings(currentSettings);
            
            // Update UI
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
    
    // Update notification handlers
    dismissUpdateButton.addEventListener('click', hideUpdateNotification);
    
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
    
    // ESC key to close modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !apiSettingsModal.classList.contains('hidden')) {
            hideAPISettingsModal();
        }
    });
    
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
}

// Initial setup
document.addEventListener('DOMContentLoaded', () => {
    // Load usage logs
    usageLogs = loadUsageLogs();
    
    initializeSettings();
    setupEventListeners();
    initializeUI();
    renderTable();
    setupInputTooltips();
    
    // Initialize update checker
    initializeUpdateChecker();
});