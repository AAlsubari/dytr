// ---------------------------------------------------------------------------
// DYTR Studio Communication Layer - Configuration Constants
// ---------------------------------------------------------------------------
// Author: Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// Licensed under the Apache License, Version 2.0
// ---------------------------------------------------------------------------

/**
 * Base URL for the FastAPI backend server.
 * Hardcoded for local development.
 */
export const API_BASE_URL = 'http://localhost:8000';

/**
 * Versioned API prefix used by all REST endpoints.
 */
export const API_PREFIX = '/api/v1';

/**
 * Full base path for REST calls: http://localhost:8000/api/v1
 */
export const API_URL = `${API_BASE_URL}${API_PREFIX}`;

/**
 * WebSocket base URL derived from the HTTP base.
 * Converts http:// to ws:// and https:// to wss://
 */
export const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

/**
 * WebSocket endpoint for live training telemetry stream.
 */
export const WS_TRAINING_URL = `${WS_BASE_URL}${API_PREFIX}/training/ws`;

// ---------------------------------------------------------------------------
// Health Check Configuration
// ---------------------------------------------------------------------------

/**
 * Maximum time (ms) to wait for the health check response
 * before falling back to simulation mode.
 */
export const HEALTH_CHECK_TIMEOUT_MS = 4000;

/**
 * Number of health check retries before giving up.
 */
export const HEALTH_CHECK_MAX_RETRIES = 1;

// ---------------------------------------------------------------------------
// WebSocket Reconnection (Exponential Backoff)
// ---------------------------------------------------------------------------

/**
 * Initial delay (ms) before the first reconnection attempt.
 */
export const WS_RECONNECT_BASE_DELAY_MS = 500;

/**
 * Maximum delay (ms) between reconnection attempts.
 * Caps the exponential growth.
 */
export const WS_RECONNECT_MAX_DELAY_MS = 16000;

/**
 * Maximum number of consecutive reconnection attempts
 * before stopping and notifying the user.
 */
export const WS_RECONNECT_MAX_ATTEMPTS = 8;

/**
 * Multiplier applied to the delay after each failed attempt.
 */
export const WS_RECONNECT_BACKOFF_FACTOR = 2;

// ---------------------------------------------------------------------------
// Simulation Configuration
// ---------------------------------------------------------------------------

/**
 * Base tick interval (ms) for simulated telemetry updates.
 * Mirrors the real server's ~200ms (5Hz) or ~100ms (10Hz) cadence.
 */
export const SIMULATION_TICK_INTERVAL_MS = 200;

/**
 * Number of simulated training steps per task stage.
 */
export const SIMULATION_STEPS_PER_STAGE = 20;

/**
 * Total number of task stages in a simulated continual learning session.
 */
export const SIMULATION_TOTAL_STAGES = 3;

// ---------------------------------------------------------------------------
// Engine Mode Constants
// ---------------------------------------------------------------------------

/**
 * Enum-like mode identifiers.
 */
export const ENGINE_MODE = Object.freeze({
  LIVE: 'live',
  LIVE_ENGINE: 'live',
  SIMULATION: 'simulation',
  INITIALIZING: 'initializing',
});

// ---------------------------------------------------------------------------
// Connection Status Constants
// ---------------------------------------------------------------------------

export const CONNECTION_STATUS = Object.freeze({
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  RECONNECTING: 'reconnecting',
  ERROR: 'error',
});
