// ---------------------------------------------------------------------------
// DyTrStudio Communication Layer - Live Engine Client (FastAPI)
// ---------------------------------------------------------------------------
// Author: Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// Licensed under the Apache License, Version 2.0
// ---------------------------------------------------------------------------
//
// Concrete implementation of IApiClient that communicates with the real
// FastAPI backend via REST (fetch) and WebSocket for live telemetry.
//
// WebSocket reconnection uses Exponential Backoff to handle mid-training
// disconnections gracefully, preserving the last received state.
// ---------------------------------------------------------------------------

import { IApiClient } from './IApiClient.js';
import {
  API_URL,
  WS_TRAINING_URL,
  WS_RECONNECT_BASE_DELAY_MS,
  WS_RECONNECT_MAX_DELAY_MS,
  WS_RECONNECT_MAX_ATTEMPTS,
  WS_RECONNECT_BACKOFF_FACTOR,
  HEALTH_CHECK_TIMEOUT_MS,
} from './config.js';

/**
 * Wraps fetch with a timeout using AbortController.
 * @param {string} url
 * @param {RequestInit} [options]
 * @param {number} [timeoutMs]
 * @returns {Promise<Response>}
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = HEALTH_CHECK_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    return response;
  } finally {
    clearTimeout(timer);
  }
}

export class FastApiClient extends IApiClient {

  constructor() {
    super();
    /** @type {WebSocket | null} */
    this._ws = null;
    /** @type {number} */
    this._reconnectAttempts = 0;
    /** @type {number | null} */
    this._reconnectTimer = null;
    /** @type {((payload: import('./types.js').TelemetryPayload) => void) | null} */
    this._telemetryCallback = null;
    /** @type {boolean} */
    this._shouldReconnect = false;
  }

  // -------------------------------------------------------------------------
  // Health & System
  // -------------------------------------------------------------------------

  async checkHealth() {
    const response = await fetchWithTimeout(`${API_URL}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
  }

  async getCapabilities() {
    const response = await fetchWithTimeout(`${API_URL}/simulations/capabilities`);
    if (!response.ok) {
      throw new Error(`Failed to fetch capabilities: ${response.status}`);
    }
    return response.json();
  }

  // -------------------------------------------------------------------------
  // Model Configuration
  // -------------------------------------------------------------------------

  async getModelPresets() {
    const response = await fetch(`${API_URL}/models/presets`);
    if (!response.ok) {
      throw new Error(`Failed to fetch presets: ${response.status}`);
    }
    return response.json();
  }

  async estimateModel(config) {
    const response = await fetch(`${API_URL}/models/estimate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!response.ok) {
      throw new Error(`Model estimation failed: ${response.status}`);
    }
    return response.json();
  }

  // -------------------------------------------------------------------------
  // Datasets
  // -------------------------------------------------------------------------

  async getDatasets() {
    const response = await fetch(`${API_URL}/datasets/prebuilt`);
    if (!response.ok) {
      throw new Error(`Failed to fetch datasets: ${response.status}`);
    }
    return response.json();
  }

  // -------------------------------------------------------------------------
  // Training
  // -------------------------------------------------------------------------

  async startTraining(config) {
    const response = await fetch(`${API_URL}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!response.ok) {
      throw new Error(`Failed to start training: ${response.status}`);
    }
    return response.json();
  }

  async stopTraining() {
    const response = await fetch(`${API_URL}/training/stop`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error(`Failed to stop training: ${response.status}`);
    }
    return response.json();
  }

  async getTrainingStatus() {
    const response = await fetch(`${API_URL}/training/status`);
    if (!response.ok) {
      throw new Error(`Failed to fetch training status: ${response.status}`);
    }
    return response.json();
  }

  /**
   * Opens a WebSocket connection to the training telemetry endpoint.
   * Implements Exponential Backoff reconnection on unexpected close.
   *
   * @param {(payload: import('./types.js').TelemetryPayload) => void} callback
   * @returns {() => void} Unsubscribe function
   */
  subscribeToTelemetry(callback) {
    this._telemetryCallback = callback;
    this._shouldReconnect = true;
    this._reconnectAttempts = 0;

    this._connectWebSocket();

    return () => {
      this._shouldReconnect = false;
      this._telemetryCallback = null;
      this._closeWebSocket();
    };
  }

  /** @private */
  _connectWebSocket() {
    this._closeWebSocket();

    try {
      this._ws = new WebSocket(WS_TRAINING_URL);
    } catch (err) {
      console.error('[FastApiClient] WebSocket construction failed:', err);
      this._scheduleReconnect();
      return;
    }

    this._ws.onopen = () => {
      console.info('[FastApiClient] WebSocket connected.');
      this._reconnectAttempts = 0;
    };

    this._ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (this._telemetryCallback) {
          this._telemetryCallback(payload);
        }
      } catch (err) {
        console.error('[FastApiClient] Failed to parse telemetry:', err);
      }
    };

    this._ws.onerror = (event) => {
      console.error('[FastApiClient] WebSocket error:', event);
    };

    this._ws.onclose = (event) => {
      console.warn('[FastApiClient] WebSocket closed:', event.code, event.reason);
      if (this._shouldReconnect) {
        this._scheduleReconnect();
      }
    };
  }

  /**
   * Schedules a reconnection attempt with exponential backoff.
   * @private
   */
  _scheduleReconnect() {
    if (this._reconnectAttempts >= WS_RECONNECT_MAX_ATTEMPTS) {
      console.error(
        `[FastApiClient] Max reconnection attempts (${WS_RECONNECT_MAX_ATTEMPTS}) reached. Giving up.`
      );
      return;
    }

    const delay = Math.min(
      WS_RECONNECT_BASE_DELAY_MS * Math.pow(WS_RECONNECT_BACKOFF_FACTOR, this._reconnectAttempts),
      WS_RECONNECT_MAX_DELAY_MS
    );

    console.info(
      `[FastApiClient] Reconnecting in ${delay}ms (attempt ${this._reconnectAttempts + 1}/${WS_RECONNECT_MAX_ATTEMPTS})...`
    );

    this._reconnectTimer = setTimeout(() => {
      this._reconnectAttempts++;
      this._connectWebSocket();
    }, delay);
  }

  /** @private */
  _closeWebSocket() {
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this._ws) {
      this._ws.onopen = null;
      this._ws.onmessage = null;
      this._ws.onerror = null;
      this._ws.onclose = null;
      if (this._ws.readyState === WebSocket.OPEN || this._ws.readyState === WebSocket.CONNECTING) {
        this._ws.close();
      }
      this._ws = null;
    }
  }

  // -------------------------------------------------------------------------
  // Inference
  // -------------------------------------------------------------------------

  async runInference(request) {
    const response = await fetch(`${API_URL}/inference/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      throw new Error(`Inference failed: ${response.status}`);
    }
    return response.json();
  }

  // -------------------------------------------------------------------------
  // Educational Simulations
  // -------------------------------------------------------------------------

  async getRopeSimulation(params = {}) {
    const query = new URLSearchParams();
    if (params.token_index !== undefined) query.set('token_index', params.token_index);
    if (params.dim !== undefined) query.set('dim', params.dim);
    if (params.base !== undefined) query.set('base', params.base);

    const response = await fetch(`${API_URL}/simulations/rope?${query}`);
    if (!response.ok) {
      throw new Error(`RoPE simulation failed: ${response.status}`);
    }
    return response.json();
  }

  async getFlashAttention(params = {}) {
    const query = new URLSearchParams();
    if (params.seq_len !== undefined) query.set('seq_len', params.seq_len);
    if (params.hidden_dim !== undefined) query.set('hidden_dim', params.hidden_dim);
    if (params.num_heads !== undefined) query.set('num_heads', params.num_heads);
    if (params.block_size !== undefined) query.set('block_size', params.block_size);

    const response = await fetch(`${API_URL}/simulations/flash-attention?${query}`);
    if (!response.ok) {
      throw new Error(`Flash attention simulation failed: ${response.status}`);
    }
    return response.json();
  }

  async getContinualLearning(params = {}) {
    const query = new URLSearchParams();
    const useEwc = params.use_ewc !== undefined ? params.use_ewc : params.use_si;
    const ewcLambda = params.ewc_lambda !== undefined ? params.ewc_lambda : params.si_c;
    if (useEwc !== undefined) query.set('use_ewc', useEwc);
    if (ewcLambda !== undefined) query.set('ewc_lambda', ewcLambda);
    if (params.use_replay !== undefined) query.set('use_replay', params.use_replay);
    if (params.num_steps !== undefined) query.set('num_steps', params.num_steps);

    const response = await fetch(`${API_URL}/simulations/continual-learning?${query}`);
    if (!response.ok) {
      throw new Error(`Continual learning simulation failed: ${response.status}`);
    }
    return response.json();
  }

  async getAdaLoRASVD(params = {}) {
    const query = new URLSearchParams();
    if (params.init_rank !== undefined) query.set('init_rank', params.init_rank);
    if (params.target_rank !== undefined) query.set('target_rank', params.target_rank);
    if (params.d_in !== undefined) query.set('d_in', params.d_in);
    if (params.d_out !== undefined) query.set('d_out', params.d_out);

    const response = await fetch(`${API_URL}/simulations/adalora-svd?${query}`);
    if (!response.ok) {
      throw new Error(`AdaLoRA simulation failed: ${response.status}`);
    }
    return response.json();
  }

  async tokenizeAlign(request = {}) {
    const response = await fetch(`${API_URL}/simulations/tokenize-align`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: request.text || 'Barack Obama visited the United Nations in New York',
        task_type: request.task_type || 'token_classification',
      }),
    });
    if (!response.ok) {
      throw new Error(`Tokenize alignment failed: ${response.status}`);
    }
    return response.json();
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  dispose() {
    this._shouldReconnect = false;
    this._telemetryCallback = null;
    this._closeWebSocket();
  }
}
