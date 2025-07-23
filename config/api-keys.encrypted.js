// Encrypted API Keys Configuration
// This file contains encrypted API keys that are automatically decrypted at runtime
// The keys are encrypted using AES-256-GCM encryption

export const ENCRYPTED_API_KEYS = {
    // Encryption salt and IV (these can be public)
    salt: 'auto-scan-app-2024',
    iv: '1234567890123456',
    
    // Encrypted API keys (Base64 encoded)
    keys: {
        // To generate these, use the encryption utility or the app's "Export Config" feature
        gemini: 'U2FsdGVkX1+vupppZksvRf5pq5g5XjFRIWpVfUh09HKLLLLtIWWoStSo5pdkQ/XfGHQ5nPhJ8pQkEJmdOW7HNr9T2r7tqsNEQXJE8N9F1P6O5MoX5MPqRVtP5MPqRVtP5MPqRVtP5MPqRVtP', // Example encrypted key
        openai: null, // Will be set when user configures
        claude: null, // Will be set when user configures  
        upstage: null // Will be set when user configures
    }
};

// Note: This is a fallback method. The primary method is still the UI-based configuration
// which stores keys in localStorage with client-side encryption.