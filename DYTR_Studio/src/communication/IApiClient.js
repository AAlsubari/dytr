// ---------------------------------------------------------------------------
// DyTrStudio Communication Layer - Abstract API Client Interface
// ---------------------------------------------------------------------------
// Author: Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// Licensed under the Apache License, Version 2.0
// ---------------------------------------------------------------------------
//
// This class defines the contract that both FastApiClient and SimulationClient
// must fulfill. Any React component or Zustand store interacts with this
// interface exclusively -- never with a concrete implementation.
//
// Every method throws by default. Subclasses MUST override all methods.
// ---------------------------------------------------------------------------

/** @typedef {import('./types.js').HealthCheckResponse}    HealthCheckResponse */
/** @typedef {import('./types.js').SystemCapabilities}     SystemCapabilities */
/** @typedef {import('./types.js').ModelPreset}            ModelPreset */
/** @typedef {import('./types.js').ModelBuildConfig}       ModelBuildConfig */
/** @typedef {import('./types.js').ModelEstimate}          ModelEstimate */
/** @typedef {import('./types.js').Dataset}                Dataset */
/** @typedef {import('./types.js').TrainingConfig}         TrainingConfig */
/** @typedef {import('./types.js').TrainingState}          TrainingState */
/** @typedef {import('./types.js').TelemetryPayload}       TelemetryPayload */
/** @typedef {import('./types.js').InferenceRequest}       InferenceRequest */
/** @typedef {import('./types.js').InferenceResult}        InferenceResult */
/** @typedef {import('./types.js').RopeData}               RopeData */
/** @typedef {import('./types.js').FlashAttentionData}     FlashAttentionData */
/** @typedef {import('./types.js').ContinualLearningData}  ContinualLearningData */
/** @typedef {import('./types.js').AdaLoRAData}            AdaLoRAData */
/** @typedef {import('./types.js').TokenAlignData}         TokenAlignData */

export class IApiClient {

  constructor() {
    if (new.target === IApiClient) {
      throw new Error('IApiClient is abstract and cannot be instantiated directly.');
    }
  }

  // -------------------------------------------------------------------------
  // Health & System
  // -------------------------------------------------------------------------

  /**
   * Check server health and dytr library availability.
   * @returns {Promise<HealthCheckResponse>}
   */
  async checkHealth() {
    throw new Error('Not implemented: checkHealth');
  }

  /**
   * Retrieve host hardware and computational capabilities.
   * @returns {Promise<SystemCapabilities>}
   */
  async getCapabilities() {
    throw new Error('Not implemented: getCapabilities');
  }

  // -------------------------------------------------------------------------
  // Model Configuration
  // -------------------------------------------------------------------------

  /**
   * Fetch pre-configured architecture presets.
   * @returns {Promise<ModelPreset[]>}
   */
  async getModelPresets() {
    throw new Error('Not implemented: getModelPresets');
  }

  /**
   * Calculate parameter breakdown and memory footprint for a given config.
   * @param {ModelBuildConfig} config
   * @returns {Promise<ModelEstimate>}
   */
  async estimateModel(config) {
    throw new Error('Not implemented: estimateModel');
  }

  // -------------------------------------------------------------------------
  // Datasets
  // -------------------------------------------------------------------------

  /**
   * Fetch available prebuilt datasets catalog.
   * @returns {Promise<Dataset[]>}
   */
  async getDatasets() {
    throw new Error('Not implemented: getDatasets');
  }

  // -------------------------------------------------------------------------
  // Training
  // -------------------------------------------------------------------------

  /**
   * Start a training session with the provided configuration.
   * @param {TrainingConfig} config
   * @returns {Promise<{ status: string, config: TrainingConfig }>}
   */
  async startTraining(config) {
    throw new Error('Not implemented: startTraining');
  }

  /**
   * Halt the currently active training session.
   * @returns {Promise<{ status: string }>}
   */
  async stopTraining() {
    throw new Error('Not implemented: stopTraining');
  }

  /**
   * Get a one-time snapshot of the current training state.
   * @returns {Promise<TrainingState>}
   */
  async getTrainingStatus() {
    throw new Error('Not implemented: getTrainingStatus');
  }

  /**
   * Subscribe to live telemetry updates (WebSocket or simulated stream).
   * Returns an unsubscribe function to stop the stream.
   *
   * @param {(payload: TelemetryPayload) => void} callback
   * @returns {() => void} Unsubscribe function
   */
  subscribeToTelemetry(callback) {
    throw new Error('Not implemented: subscribeToTelemetry');
  }

  // -------------------------------------------------------------------------
  // Inference
  // -------------------------------------------------------------------------

  /**
   * Run model prediction on input text.
   * @param {InferenceRequest} request
   * @returns {Promise<InferenceResult>}
   */
  async runInference(request) {
    throw new Error('Not implemented: runInference');
  }

  // -------------------------------------------------------------------------
  // Educational Simulations (Glassroom)
  // -------------------------------------------------------------------------

  /**
   * Simulate Rotary Position Embedding (RoPE) 2D rotation.
   * @param {{ token_index?: number, dim?: number, base?: number }} params
   * @returns {Promise<RopeData>}
   */
  async getRopeSimulation(params = {}) {
    throw new Error('Not implemented: getRopeSimulation');
  }

  /**
   * Simulate FlashAttention SRAM vs HBM memory statistics.
   * @param {{ seq_len?: number, hidden_dim?: number, num_heads?: number, block_size?: number }} params
   * @returns {Promise<FlashAttentionData>}
   */
  async getFlashAttention(params = {}) {
    throw new Error('Not implemented: getFlashAttention');
  }

  /**
   * Simulate 2-task continual learning with/without Elastic Weight Consolidation (EWC).
   * @param {{ use_ewc?: boolean, ewc_lambda?: number, use_replay?: boolean, num_steps?: number }} params
   * @returns {Promise<ContinualLearningData>}
   */
  async getContinualLearning(params = {}) {
    throw new Error('Not implemented: getContinualLearning');
  }

  /**
   * Simulate AdaLoRA SVD rank decomposition and pruning.
   * @param {{ init_rank?: number, target_rank?: number, d_in?: number, d_out?: number }} params
   * @returns {Promise<AdaLoRAData>}
   */
  async getAdaLoRASVD(params = {}) {
    throw new Error('Not implemented: getAdaLoRASVD');
  }

  /**
   * Simulate subword tokenization and -100 label alignment.
   * @param {{ text?: string, task_type?: string }} request
   * @returns {Promise<TokenAlignData>}
   */
  async tokenizeAlign(request = {}) {
    throw new Error('Not implemented: tokenizeAlign');
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /**
   * Clean up connections, intervals, and WebSocket handles.
   * Called when the client is being replaced or the app unmounts.
   */
  dispose() {
    // Default no-op. Override in subclasses that hold open resources.
  }
}
