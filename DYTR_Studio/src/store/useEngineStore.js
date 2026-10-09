// ---------------------------------------------------------------------------
// DYTR Studio - Central Engine Store (Zustand)
// ---------------------------------------------------------------------------
// Author: Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// Licensed under the Apache License, Version 2.0
// ---------------------------------------------------------------------------
//
// Single source of truth for all communication state.
// React components subscribe to this store -- never call the API client
// directly. The store handles initialization, mode switching, telemetry
// subscriptions, and reconnection.
// ---------------------------------------------------------------------------

import { create } from 'zustand';
import { resolveApiClient, attemptReconnect } from '../communication/ApiClientFactory.js';
import { ENGINE_MODE, CONNECTION_STATUS } from '../communication/config.js';

/**
 * @typedef {import('../communication/IApiClient.js').IApiClient} IApiClient
 * @typedef {import('../communication/types.js').TelemetryPayload} TelemetryPayload
 * @typedef {import('../communication/types.js').SystemCapabilities} SystemCapabilities
 */

const useEngineStore = create((set, get) => ({

  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------

  /** @type {IApiClient | null} */
  client: null,

  /** @type {string} */
  engineMode: ENGINE_MODE.INITIALIZING,

  /** @type {string} */
  connectionStatus: CONNECTION_STATUS.DISCONNECTED,

  /** @type {string | null} */
  fallbackReason: null,

  /** @type {boolean} */
  isInitialized: false,

  /** @type {string | null} */
  notification: null,

  /** @type {TelemetryPayload | null} */
  latestTelemetry: null,

  /** @type {SystemCapabilities | null} */
  capabilities: null,

  /** @type {boolean} */
  isReconnecting: false,

  /** @type {(() => void) | null} */
  _telemetryUnsubscribe: null,

  // -------------------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------------------

  /**
   * Bootstrap the communication layer.
   * Called once when the application mounts.
   */
  initialize: async () => {
    const state = get();
    if (state.isInitialized) return;

    try {
      const { client, mode, reason } = await resolveApiClient();

      const updates = {
        client,
        engineMode: mode,
        isInitialized: true,
        fallbackReason: reason,
        connectionStatus: mode === ENGINE_MODE.LIVE
          ? CONNECTION_STATUS.CONNECTED
          : CONNECTION_STATUS.DISCONNECTED,
        notification: null,
      };

      if (mode === ENGINE_MODE.SIMULATION && reason) {
        updates.notification = 'System operating in Educational Simulation Mode due to server unavailability.';
      }

      set(updates);

      // Fetch capabilities after initialization
      try {
        const caps = await client.getCapabilities();
        set({ capabilities: caps });
      } catch (err) {
        console.error('[EngineStore] Failed to fetch capabilities:', err);
      }
    } catch (error) {
      console.error('[EngineStore] Initialization failed:', error);
      set({
        isInitialized: true,
        engineMode: ENGINE_MODE.SIMULATION,
        connectionStatus: CONNECTION_STATUS.ERROR,
        notification: 'System initialization failed. Operating in Simulation Mode.',
        fallbackReason: error.message,
      });
    }
  },

  // -------------------------------------------------------------------------
  // Reconnection (Manual Button)
  // -------------------------------------------------------------------------

  /**
   * Attempt to reconnect to the live server.
   * Disposes the current client and resolves a new one.
   */
  reconnect: async () => {
    const state = get();
    if (state.isReconnecting) return;

    set({ isReconnecting: true, notification: null });

    // Unsubscribe from any active telemetry stream
    if (state._telemetryUnsubscribe) {
      state._telemetryUnsubscribe();
    }

    try {
      const { client, mode, reason } = await attemptReconnect(state.client);

      const updates = {
        client,
        engineMode: mode,
        fallbackReason: reason,
        isReconnecting: false,
        _telemetryUnsubscribe: null,
        latestTelemetry: null,
        connectionStatus: mode === ENGINE_MODE.LIVE
          ? CONNECTION_STATUS.CONNECTED
          : CONNECTION_STATUS.DISCONNECTED,
        notification: null,
      };

      if (mode === ENGINE_MODE.LIVE) {
        updates.notification = 'Successfully connected to the live DyTR engine.';
      } else if (reason) {
        updates.notification = 'Reconnection failed. Remaining in Simulation Mode.';
      }

      set(updates);

      // Refresh capabilities
      try {
        const caps = await client.getCapabilities();
        set({ capabilities: caps });
      } catch (err) {
        console.error('[EngineStore] Failed to refresh capabilities:', err);
      }
    } catch (error) {
      console.error('[EngineStore] Reconnection error:', error);
      set({
        isReconnecting: false,
        notification: 'Reconnection failed: ' + error.message,
      });
    }
  },

  // -------------------------------------------------------------------------
  // Telemetry Subscription
  // -------------------------------------------------------------------------

  /**
   * Start subscribing to training telemetry from the active client.
   * The latest payload is stored in state for all subscribers.
   */
  subscribeTelemetry: () => {
    const state = get();
    if (!state.client) return;

    // Clean up existing subscription
    if (state._telemetryUnsubscribe) {
      state._telemetryUnsubscribe();
    }

    const unsubscribe = state.client.subscribeToTelemetry((payload) => {
      set({ latestTelemetry: payload });
    });

    set({ _telemetryUnsubscribe: unsubscribe });
  },

  /**
   * Stop the active telemetry subscription.
   */
  unsubscribeTelemetry: () => {
    const state = get();
    if (state._telemetryUnsubscribe) {
      state._telemetryUnsubscribe();
      set({ _telemetryUnsubscribe: null });
    }
  },

  // -------------------------------------------------------------------------
  // Notification Management
  // -------------------------------------------------------------------------

  /**
   * Clear the current UI notification.
   */
  clearNotification: () => {
    set({ notification: null });
  },

  // -------------------------------------------------------------------------
  // Cleanup
  // -------------------------------------------------------------------------

  /**
   * Dispose the client and reset state.
   * Called when the application unmounts.
   */
  dispose: () => {
    const state = get();
    if (state._telemetryUnsubscribe) {
      state._telemetryUnsubscribe();
    }
    if (state.client) {
      state.client.dispose();
    }
    set({
      client: null,
      engineMode: ENGINE_MODE.INITIALIZING,
      connectionStatus: CONNECTION_STATUS.DISCONNECTED,
      isInitialized: false,
      latestTelemetry: null,
      _telemetryUnsubscribe: null,
      notification: null,
    });
  },
}));

export default useEngineStore;
