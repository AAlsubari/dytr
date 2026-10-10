// ---------------------------------------------------------------------------
// DyTrStudio Communication Layer - API Client Factory
// ---------------------------------------------------------------------------
// Author: Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// Licensed under the Apache License, Version 2.0
// ---------------------------------------------------------------------------
//
// Resolves the correct client (FastApiClient or SimulationClient) based on
// a health check ping at application initialization.
//
// Decision logic:
//   1. Ping the server at /api/v1/health with a timeout.
//   2. If the server responds AND reports dytr_available === true
//      --> return FastApiClient (Live Engine Mode).
//   3. If the connection fails, times out, or dytr_available === false
//      --> return SimulationClient (Simulation Mode).
// ---------------------------------------------------------------------------

import { FastApiClient } from './FastApiClient.js';
import { SimulationClient } from './SimulationClient.js';
import { ENGINE_MODE } from './config.js';

/**
 * @typedef {Object} FactoryResult
 * @property {import('./IApiClient.js').IApiClient} client - The resolved API client
 * @property {string} mode - 'live' | 'simulation'
 * @property {string | null} reason - Explanation when falling back to simulation
 */

/**
 * Performs the health check and instantiates the appropriate client.
 * This function should be called once during application bootstrap.
 *
 * @returns {Promise<FactoryResult>}
 */
export async function resolveApiClient() {
  const liveClient = new FastApiClient();

  try {
    const health = await liveClient.checkHealth();

    // Server is reachable -- check if dytr library is loaded
    if (health && health.dytr_available === true) {
      return {
        client: liveClient,
        mode: ENGINE_MODE.LIVE,
        reason: null,
      };
    }

    // Server is reachable but dytr is not available
    liveClient.dispose();
    return {
      client: new SimulationClient(),
      mode: ENGINE_MODE.SIMULATION,
      reason: 'Server is online but the DyTR library is not installed or failed to load.',
    };
  } catch (error) {
    // Server unreachable or timed out
    liveClient.dispose();
    return {
      client: new SimulationClient(),
      mode: ENGINE_MODE.SIMULATION,
      reason: `Cannot reach the backend server: ${error.message || 'connection failed'}.`,
    };
  }
}

/**
 * Manually attempts reconnection to the live server.
 * Called when the user clicks the "Reconnect to Server" button.
 *
 * @param {import('./IApiClient.js').IApiClient} currentClient - The active client to dispose if switching
 * @returns {Promise<FactoryResult>}
 */
export async function attemptReconnect(currentClient) {
  if (currentClient) {
    currentClient.dispose();
  }
  return resolveApiClient();
}
