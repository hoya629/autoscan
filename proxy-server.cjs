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
            : 'https://api.upstage.ai/v1/document-ai/document-parse';

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
            // Document Parse API uses multipart/form-data
            const FormData = require('form-data');
            const formData = new FormData();
            
            console.log('🔍 [Upstage Proxy] Processing document data...');
            console.log('🔍 [Upstage Proxy] Full request body:', JSON.stringify(requestBody, null, 2));
            
            // Extract document data from base64
            if (requestBody.document && requestBody.document.startsWith('data:')) {
                const [header, base64Data] = requestBody.document.split(',');
                const mimeType = header.match(/data:([^;]+)/)?.[1] || 'image/jpeg';
                const buffer = Buffer.from(base64Data, 'base64');
                
                console.log('🔍 [Upstage Proxy] Document info:', {
                    mimeType,
                    bufferSize: buffer.length,
                    base64Length: base64Data.length,
                    headerInfo: header
                });
                
                // Use the field name 'document' as expected by the API
                // Specify proper file options with contentType for multipart
                const extension = mimeType.includes('pdf') ? 'pdf' : 
                                 mimeType.includes('png') ? 'png' : 
                                 mimeType.includes('jpeg') || mimeType.includes('jpg') ? 'jpg' : 'jpg';
                const filename = `document.${extension}`;
                
                formData.append('document', buffer, {
                    filename: filename,
                    contentType: mimeType,
                    knownLength: buffer.length
                });
                
                console.log('📤 [Upstage Proxy] FormData field added - document');
            } else if (requestBody.document) {
                console.log('⚠️ [Upstage Proxy] Document format not recognized:', requestBody.document.substring(0, 50));
                throw new Error('Document data must be in base64 data URL format');
            } else {
                console.log('❌ [Upstage Proxy] No document field found in request');
                throw new Error('Document field is required');
            }
            
            // Add Upstage Document Parse API specific parameters based on REAL working format
            // Model parameter (required)
            formData.append('model', 'document-parse');
            console.log('📤 [Upstage Proxy] FormData field added - model: document-parse');
            
            // OCR parameter: 'auto' is the correct value from working example
            formData.append('ocr', 'auto');
            console.log('📤 [Upstage Proxy] FormData field added - ocr: auto');
            
            // Output formats: Only text format as per working example
            formData.append('output_formats', '["text"]');
            console.log('📤 [Upstage Proxy] FormData field added - output_formats: ["text"]');
            
            // Log the complete form data structure
            console.log('🚀 [Upstage Proxy] Sending request to:', apiEndpoint);
            console.log('🚀 [Upstage Proxy] Request headers will include:', {
                'Authorization': `Bearer ${apiKey.substring(0, 10)}...`,
                'Content-Type': formData.getHeaders()['content-type']
            });
            
            try {
                response = await axios.post(apiEndpoint, formData, {
                    headers: {
                        'Authorization': `Bearer ${apiKey}`,
                        ...formData.getHeaders()
                    },
                    timeout: 120000, // 2 minutes timeout for document processing
                    maxBodyLength: Infinity,
                    maxContentLength: Infinity
                });
                
                console.log('✅ [Upstage Proxy] Response received:', {
                    status: response.status,
                    statusText: response.statusText,
                    contentType: response.headers['content-type'],
                    dataKeys: response.data ? Object.keys(response.data) : 'no data'
                });
            } catch (axiosError) {
                console.error('❌ [Upstage Proxy] Axios error details:', {
                    status: axiosError.response?.status,
                    statusText: axiosError.response?.statusText,
                    headers: axiosError.response?.headers,
                    data: axiosError.response?.data,
                    message: axiosError.message
                });
                throw axiosError;
            }
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