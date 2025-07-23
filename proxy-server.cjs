const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = 3002;

// CORS 설정
app.use(cors({
    origin: ['http://localhost:3000', 'http://localhost:3001'],
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(express.json({ limit: '50mb' }));

// Claude API 프록시
app.post('/api/claude', async (req, res) => {
    try {
        // 1순위: 요청에서 전달된 API 키
        const apiKey = req.body.apiKey || req.headers['x-api-key'] || process.env.VITE_CLAUDE_API_KEY;
        console.log('Claude API Key source:', req.body.apiKey ? 'request body' : req.headers['x-api-key'] ? 'header' : 'env');
        console.log('Claude API Key length:', apiKey?.length || 0);
        console.log('Claude API Key prefix:', apiKey?.substring(0, 10) || 'none');
        
        if (!apiKey || apiKey.includes('input_your_api_key')) {
            return res.status(400).json({ error: 'Claude API key not configured' });
        }

        // Map model names to actual Claude API model names
        const modelMapping = {
            'claude-sonnet-4-20250514': 'claude-sonnet-4-20250514',
            'claude-opus-4-20250514': 'claude-opus-4-20250514', 
            'claude-3-5-sonnet-20241022': 'claude-3-5-sonnet-20241022'
        };
        
        const actualModel = modelMapping[req.body.model] || 'claude-sonnet-4-20250514';
        
        const response = await axios.post('https://api.anthropic.com/v1/messages', {
            model: actualModel,
            max_tokens: 4000,
            messages: req.body.messages
        }, {
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': apiKey,
                'anthropic-version': '2023-06-01'
            }
        });

        res.json(response.data);
    } catch (error) {
        console.error('Claude API error:', error.response?.data || error.message);
        res.status(error.response?.status || 500).json({
            error: `Claude API error: ${error.response?.status || 500}`,
            details: error.response?.data?.error?.message || error.message
        });
    }
});

// OpenAI API 프록시
app.post('/api/openai', async (req, res) => {
    try {
        // 1순위: 요청에서 전달된 API 키
        const apiKey = req.body.apiKey || req.headers['x-api-key'] || process.env.OPENAI_API_KEY;
        console.log('OpenAI API Key source:', req.body.apiKey ? 'request body' : req.headers['x-api-key'] ? 'header' : 'env');
        console.log('OpenAI API Key length:', apiKey?.length || 0);
        console.log('OpenAI API Key prefix:', apiKey?.substring(0, 10) || 'none');
        
        if (!apiKey || apiKey.includes('input_your_api_key')) {
            return res.status(400).json({ error: 'OpenAI API key not configured' });
        }

        // API 키를 요청 본문에서 제거하고 나머지 데이터만 전달
        const { apiKey: _, ...requestBody } = req.body;
        
        const response = await axios.post('https://api.openai.com/v1/chat/completions', requestBody, {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            }
        });

        res.json(response.data);
    } catch (error) {
        console.error('OpenAI API error:', error.response?.data || error.message);
        res.status(error.response?.status || 500).json({
            error: `OpenAI API error: ${error.response?.status || 500}`,
            details: error.response?.data?.error?.message || error.message
        });
    }
});

// Gemini API 프록시
app.post('/api/gemini', async (req, res) => {
    try {
        // 1순위: 요청에서 전달된 API 키
        const apiKey = req.body.apiKey || req.headers['x-api-key'] || process.env.GEMINI_API_KEY;
        console.log('Gemini API Key source:', req.body.apiKey ? 'request body' : req.headers['x-api-key'] ? 'header' : 'env');
        console.log('Gemini API Key length:', apiKey?.length || 0);
        console.log('Gemini API Key prefix:', apiKey?.substring(0, 10) || 'none');
        
        if (!apiKey || apiKey.includes('input_your_api_key')) {
            return res.status(400).json({ error: 'Gemini API key not configured' });
        }

        // Gemini API endpoint
        const modelName = req.body.model || 'gemini-2.5-flash';
        const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`;
        
        // API 키를 요청 본문에서 제거하고 나머지 데이터만 전달
        const { apiKey: _, ...requestBody } = req.body;
        
        const response = await axios.post(apiUrl, {
            contents: requestBody.contents,
            generationConfig: requestBody.config || {}
        }, {
            headers: {
                'Content-Type': 'application/json'
            }
        });

        res.json(response.data);
    } catch (error) {
        console.error('Gemini API error:', error.response?.data || error.message);
        res.status(error.response?.status || 500).json({
            error: `Gemini API error: ${error.response?.status || 500}`,
            details: error.response?.data?.error?.message || error.message
        });
    }
});

// Upstage API 프록시
app.post('/api/upstage', async (req, res) => {
    try {
        // 1순위: 요청에서 전달된 API 키
        const apiKey = req.body.apiKey || req.headers['x-api-key'] || process.env.UPSTAGE_API_KEY;
        console.log('Upstage API Key source:', req.body.apiKey ? 'request body' : req.headers['x-api-key'] ? 'header' : 'env');
        console.log('Upstage API Key length:', apiKey?.length || 0);
        console.log('Upstage API Key prefix:', apiKey?.substring(0, 10) || 'none');
        
        if (!apiKey || apiKey.includes('input_your_api_key')) {
            return res.status(400).json({ error: 'Upstage API key not configured' });
        }

        // API 키를 요청 본문에서 제거하고 나머지 데이터만 전달
        const { apiKey: _, ...requestBody } = req.body;
        
        // Determine API endpoint based on model
        const isDocVision = requestBody.model === 'solar-docvision-preview';
        const apiEndpoint = isDocVision 
            ? 'https://api.upstage.ai/v1/solar/chat/completions'
            : 'https://api.upstage.ai/v1/document-digitization';

        let response;
        if (isDocVision) {
            // Solar DocVision uses chat completions format
            response = await axios.post(apiEndpoint, requestBody, {
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                }
            });
        } else {
            // Document Parse API uses form data
            response = await axios.post(apiEndpoint, requestBody, {
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': req.headers['content-type'] || 'application/json'
                }
            });
        }

        res.json(response.data);
    } catch (error) {
        console.error('Upstage API error:', error.response?.data || error.message);
        res.status(error.response?.status || 500).json({
            error: `Upstage API error: ${error.response?.status || 500}`,
            details: error.response?.data?.error?.message || error.message
        });
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
    console.log(`Proxy server running on http://localhost:${PORT}`);
    console.log('Available endpoints:');
    console.log('- POST /api/gemini - Gemini API proxy');
    console.log('- POST /api/openai - OpenAI API proxy');
    console.log('- POST /api/upstage - Upstage API proxy');
    console.log('- GET /health - Health check');
});