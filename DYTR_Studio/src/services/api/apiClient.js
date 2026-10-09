// ==============================================================================
// DYTR Studio Core API Client Layer
// Provides reliable, decoupled HTTP communication with the Backend / Mock Server
// ==============================================================================

const getDynamicApiBase = () => {
  if (typeof window === 'undefined') return 'http://localhost:8000/api/v1';
  if (window.__DYTR_API_URL__) return window.__DYTR_API_URL__;
  const host = window.location.hostname || 'localhost';
  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
  return `${protocol}//${host}:8000/api/v1`;
};

const getDynamicMockBase = () => {
  if (typeof window === 'undefined') return 'http://localhost:3001/api/v1';
  const host = window.location.hostname || 'localhost';
  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
  return `${protocol}//${host}:3001/api/v1`;
};

class ApiClient {
  constructor() {
    this.getPrimaryUrl = () => getDynamicApiBase();
    this.getFallbackUrl = () => getDynamicMockBase();
  }

  /**
   * Performs an HTTP request with automatic timeout and fallback handling.
   */
  async request(endpoint, options = {}) {
    const { timeout = 6000, ...customConfig } = options;

    const executeFetch = async (baseUrl) => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      try {
        const url = `${baseUrl}${endpoint}`;
        const config = {
          method: options.method || 'GET',
          headers: {
            'Content-Type': 'application/json',
            ...(customConfig.headers || {}),
          },
          signal: controller.signal,
          ...customConfig,
        };

        const res = await fetch(url, config);
        clearTimeout(timeoutId);

        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.detail || errBody.message || `HTTP ${res.status}`);
        }

        return await res.json();
      } catch (err) {
        clearTimeout(timeoutId);
        throw err;
      }
    };

    // 1. Try Primary API (FastAPI backend)
    try {
      return await executeFetch(this.getPrimaryUrl());
    } catch (primaryErr) {
      // 2. Try Secondary Mock API server (Port 3001)
      try {
        return await executeFetch(this.getFallbackUrl());
      } catch (fallbackErr) {
        // Bubble up error to service layer for graceful domain fallback
        throw new Error(`API Unreachable: [Primary: ${primaryErr.message}] [Mock: ${fallbackErr.message}]`);
      }
    }
  }

  get(endpoint, options) {
    return this.request(endpoint, { ...options, method: 'GET' });
  }

  post(endpoint, body, options) {
    return this.request(endpoint, {
      ...options,
      method: 'POST',
      body: JSON.stringify(body),
    });
  }
}

export const apiClient = new ApiClient();
