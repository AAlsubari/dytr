// ---------------------------------------------------------------------------
// DyTrStudio Communication Layer - Simulation Client (Mock Engine)
// ---------------------------------------------------------------------------
// Author: Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// Licensed under the Apache License, Version 2.0
// ---------------------------------------------------------------------------
//
// Client-side mock engine that generates realistic training data without
// any network dependency. Uses delayed promises and constrained random
// generators to simulate model training behavior, loss curves, VRAM usage,
// and R-Matrix updates.
//
// Data schemas are 100% identical to what FastApiClient returns from the
// real server. React components remain completely agnostic.
// ---------------------------------------------------------------------------

import { IApiClient } from './IApiClient.js';
import {
  SIMULATION_TICK_INTERVAL_MS,
  SIMULATION_STEPS_PER_STAGE,
  SIMULATION_TOTAL_STAGES,
} from './config.js';

// ---------------------------------------------------------------------------
// Utility: constrained random number within a range
// ---------------------------------------------------------------------------

function randomInRange(min, max) {
  return min + Math.random() * (max - min);
}

function roundTo(value, decimals) {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
}

/**
 * Returns a promise that resolves after a short simulated delay.
 * Mimics network latency without actual network calls.
 * @param {number} [ms] - Delay in milliseconds (default: 80-200ms)
 */
function simulatedDelay(ms) {
  const delay = ms !== undefined ? ms : Math.floor(randomInRange(80, 200));
  return new Promise((resolve) => setTimeout(resolve, delay));
}

// ---------------------------------------------------------------------------
// SimulationClient
// ---------------------------------------------------------------------------

export class SimulationClient extends IApiClient {

  constructor() {
    super();
    /** @type {number | null} */
    this._telemetryInterval = null;
    /** @type {boolean} */
    this._isTraining = false;
    /** @type {number} */
    this._currentStage = 0;
    /** @type {number} */
    this._currentStep = 0;
    /** @type {number[][]} */
    this._rMatrix = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ];
    /** @type {TrainingConfig | null} */
    this._trainingConfig = null;
  }

  // -------------------------------------------------------------------------
  // Health & System
  // -------------------------------------------------------------------------

  async checkHealth() {
    await simulatedDelay(100);
    return {
      status: 'online',
      service: 'DyTrStudio Simulation Engine',
      version: '0.1.3',
      dytr_available: false,
      engine_mode: 'simulation',
    };
  }

  async getCapabilities() {
    await simulatedDelay();
    return {
      dytr_version: '0.1.3 (simulated)',
      dytr_available: false,
      cuda_available: false,
      device_name: 'Simulation Engine (No GPU)',
      vram_total_gb: 0.0,
      flash_attention_supported: false,
      amp_supported: false,
      bf16_supported: false,
      torch_version: 'N/A (simulation)',
    };
  }

  // -------------------------------------------------------------------------
  // Model Configuration
  // -------------------------------------------------------------------------

  async getModelPresets() {
    await simulatedDelay();
    return [
      {
        id: 'sentiment_analyst',
        name: 'Smart Sentiment Analyst',
        description: 'Lightweight sequence classification model optimized for emotional tone and polarity.',
        category: 'Classification',
        badge: 'Fast & Efficient',
        config: {
          embed_dim: 128,
          num_layers: 2,
          num_heads: 4,
          head_dim: 32,
          max_seq_len: 128,
          use_flash_attention: true,
          use_rotary_embedding: true,
          use_amp: true,
          use_si: true,
          tasks: [
            { task_name: 'task_sentiment', task_type: 'sentence_classification', num_classes: 2, max_len: 64 },
          ],
        },
      },
      {
        id: 'multi_task_champion',
        name: 'Multi-Task Continual Champion',
        description: 'Full Continual Learning benchmark model with Sentiment, Topic, and NER heads protected by Elastic Weight Consolidation (EWC).',
        category: 'Continual Learning',
        badge: 'Arena Ready',
        config: {
          embed_dim: 256,
          num_layers: 4,
          num_heads: 4,
          head_dim: 64,
          max_seq_len: 256,
          use_flash_attention: true,
          use_rotary_embedding: true,
          use_amp: true,
          use_si: true,
          si_c: 1.0,
          tasks: [
            { task_name: 'task_sentiment', task_type: 'sentence_classification', num_classes: 2, max_len: 64 },
            { task_name: 'task_topic', task_type: 'sentence_classification', num_classes: 4, max_len: 64 },
            { task_name: 'task_ner', task_type: 'token_classification', num_classes: 5, max_len: 64 },
          ],
        },
      },
      {
        id: 'adalora_lightweight',
        name: 'AdaLoRA Adaptive PEFT Model',
        description: 'Memory-efficient transformer utilizing adaptive SVD low-rank updates and sensitivity budget shrinking.',
        category: 'Parameter Efficient',
        badge: 'Ultra Low VRAM',
        config: {
          embed_dim: 256,
          num_layers: 4,
          num_heads: 4,
          head_dim: 64,
          max_seq_len: 256,
          use_flash_attention: true,
          use_rotary_embedding: true,
          use_adalora: true,
          adalora_init_rank: 16,
          adalora_target_rank: 4,
          tasks: [
            { task_name: 'task_topic', task_type: 'sentence_classification', num_classes: 4, max_len: 64 },
          ],
        },
      },
      {
        id: 'neural_storyteller',
        name: 'Neural Storyteller (Causal LM)',
        description: 'Autoregressive generative language model with causal attention masking and sliding window processing.',
        category: 'Generative',
        badge: 'Text Generation',
        config: {
          embed_dim: 384,
          num_layers: 6,
          num_heads: 6,
          head_dim: 64,
          max_seq_len: 512,
          use_flash_attention: true,
          use_rotary_embedding: true,
          use_amp: true,
          tasks: [
            { task_name: 'task_story_generation', task_type: 'causal_lm', num_classes: 30522, max_len: 256 },
          ],
        },
      },
    ];
  }

  async estimateModel(config) {
    await simulatedDelay();

    const vocabSize = config.vocab_size || 30522;
    const embedDim = config.embed_dim || 256;
    const numLayers = config.num_layers || 4;
    const numHeads = config.num_heads || 4;
    const ffMult = config.ff_mult || 4;
    const useRoPE = config.use_rotary_embedding !== false;
    const maxSeqLen = config.max_seq_len || 256;
    const tasks = config.tasks || [];

    const embedParams = vocabSize * embedDim;
    const posParams = useRoPE ? 0 : maxSeqLen * embedDim;

    // Per layer: MHA(4 * d^2 + 4d) + FFN(2 * d * d*ff + d + d*ff) + Norms(4d)
    const mhaParams = (4 * embedDim * embedDim) + (4 * embedDim);
    const ffnDim = embedDim * ffMult;
    const ffnParams = (2 * embedDim * ffnDim) + embedDim + ffnDim;
    const normParams = 4 * embedDim;
    const layerParams = mhaParams + ffnParams + normParams;
    const encoderParams = numLayers * layerParams;

    const headBreakdowns = [];
    let totalHeadParams = 0;
    for (const task of tasks) {
      let hParams;
      if (task.task_type === 'causal_lm') {
        hParams = embedDim * vocabSize;
      } else if (task.task_type === 'sentence_classification' || task.task_type === 'token_classification') {
        const numC = task.num_classes || 2;
        hParams = (embedDim * numC) + numC;
      } else {
        hParams = embedDim * 64;
      }
      totalHeadParams += hParams;
      headBreakdowns.push({
        task_name: task.task_name,
        task_type: task.task_type,
        params: hParams,
      });
    }

    const totalParams = embedParams + posParams + encoderParams + totalHeadParams;
    const fp32Mb = roundTo((totalParams * 4) / (1024 * 1024), 2);
    const fp16Mb = roundTo((totalParams * 2) / (1024 * 1024), 2);

    return {
      total_parameters: totalParams,
      total_parameters_formatted: totalParams.toLocaleString(),
      embedding_params: embedParams,
      encoder_params: encoderParams,
      head_params_total: totalHeadParams,
      head_breakdowns: headBreakdowns,
      memory_footprint: {
        fp32_mb: fp32Mb,
        fp16_amp_mb: fp16Mb,
      },
      python_code: '# [Simulation Mode] Python code generation requires a live server connection.',
    };
  }

  // -------------------------------------------------------------------------
  // Datasets
  // -------------------------------------------------------------------------

  async getDatasets() {
    await simulatedDelay();
    return [
      {
        id: 'imdb_sentiment',
        name: 'IMDb Movie Sentiment',
        task_name: 'task_sentiment',
        task_type: 'sentence_classification',
        num_samples: 5000,
        classes: ['Negative (0)', 'Positive (1)'],
        description: 'Binary emotional tone classification on movie reviews.',
        sample_rows: [
          { text: 'This movie was an absolute masterpiece with brilliant acting.', label: 1 },
          { text: 'The plot was terrible, predictable and completely boring.', label: 0 },
          { text: 'Incredible directing and breathtaking cinematography throughout.', label: 1 },
          { text: 'Waste of time, poor script and unconvincing characters.', label: 0 },
        ],
      },
      {
        id: 'ag_news_topics',
        name: 'AG_News Topic Categorization',
        task_name: 'task_topic',
        task_type: 'sentence_classification',
        num_samples: 8000,
        classes: ['World (0)', 'Sports (1)', 'Business (2)', 'Sci/Tech (3)'],
        description: '4-way classification of global news headlines and thematic articles.',
        sample_rows: [
          { text: 'Global summit leaders meet in Geneva to negotiate historic climate treaty.', label: 0 },
          { text: 'Championship final goes to overtime as the point guard scores game winner.', label: 1 },
          { text: 'Central bank raises interest rates amid rising quarterly inflation figures.', label: 2 },
          { text: 'Artificial intelligence researchers develop revolutionary quantum neural chip.', label: 3 },
        ],
      },
      {
        id: 'conll_ner',
        name: 'CoNLL-2003 Named Entity Recognition',
        task_name: 'task_ner',
        task_type: 'token_classification',
        num_samples: 3500,
        classes: ['O (0)', 'B-PER (1)', 'I-PER (2)', 'B-ORG (3)', 'B-LOC (4)'],
        description: 'Microscopic token-level extraction of Person, Organization, and Location entities.',
        sample_rows: [
          {
            text: 'Barack Obama visited the United Nations in New York',
            tags: '1,2,0,0,3,3,0,4,4',
            annotated_preview: '[Barack Obama](PER) visited the [United Nations](ORG) in [New York](LOC)',
          },
          {
            text: 'Apple and Google announced joint technological initiative across Europe',
            tags: '3,0,3,0,0,0,0,4',
            annotated_preview: '[Apple](ORG) and [Google](ORG) announced joint initiative across [Europe](LOC)',
          },
        ],
      },
    ];
  }

  // -------------------------------------------------------------------------
  // Training
  // -------------------------------------------------------------------------

  async startTraining(config) {
    await simulatedDelay();
    this._trainingConfig = config;
    this._isTraining = true;
    this._currentStage = 0;
    this._currentStep = 0;
    this._rMatrix = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ];
    return { status: 'started', config };
  }

  async stopTraining() {
    await simulatedDelay(50);
    this._isTraining = false;
    return { status: 'stopped' };
  }

  async getTrainingStatus() {
    await simulatedDelay(50);
    return {
      is_training: this._isTraining,
      current_stage: this._currentStage + 1,
      total_stages: SIMULATION_TOTAL_STAGES,
      current_task: this._getTaskKey(this._currentStage),
      step: this._currentStep,
      total_steps: SIMULATION_STEPS_PER_STAGE,
      loss: 0.0,
      vram_mb: 0,
      tokens_per_sec: 0,
      r_matrix: this._rMatrix.map((row) => [...row]),
      metrics: { acc: 0.0, bwt: 0.0, fwt: 0.0 },
    };
  }

  /**
   * Starts a simulated telemetry stream via setInterval.
   * Mirrors the real WebSocket cadence (~200ms ticks).
   *
   * @param {(payload: import('./types.js').TelemetryPayload) => void} callback
   * @returns {() => void} Unsubscribe function
   */
  subscribeToTelemetry(callback) {
    this._clearTelemetryInterval();

    const tasks = [
      { key: 'task_sentiment', label: 'Sentiment Analysis (IMDb)' },
      { key: 'task_topic', label: 'Topic Classification (AG_News)' },
      { key: 'task_ner', label: 'Named Entity Recognition (CoNLL)' },
    ];

    this._telemetryInterval = setInterval(() => {
      if (!this._isTraining) return;

      this._currentStep++;
      const stageIdx = this._currentStage;
      const task = tasks[stageIdx] || tasks[0];
      const progress = this._currentStep / SIMULATION_STEPS_PER_STAGE;

      // Synthetic loss: exponential decay with jitter
      const baseLoss = stageIdx === 0 ? 1.8 : stageIdx === 1 ? 1.4 : 1.6;
      const currentLoss = Math.max(
        0.05,
        roundTo(baseLoss * Math.exp(-2.5 * progress) + randomInRange(-0.03, 0.03), 4)
      );

      // Simulated VRAM and throughput
      const vram = 2400 + Math.floor(350 * Math.sin(this._currentStep * 0.2)) + (stageIdx * 150);
      const tokensPerSec = Math.floor(randomInRange(4200, 5800));

      // Stage completion: update R-Matrix
      if (this._currentStep >= SIMULATION_STEPS_PER_STAGE) {
        // Diagonal: instantaneous accuracy
        this._rMatrix[stageIdx][stageIdx] = roundTo(randomInRange(0.78, 0.88), 4);

        // Below diagonal: retrospective recall (with SI protection)
        for (let prev = 0; prev < stageIdx; prev++) {
          const useSi = this._trainingConfig?.use_si !== false;
          if (useSi) {
            this._rMatrix[stageIdx][prev] = roundTo(
              this._rMatrix[prev][prev] - randomInRange(0.01, 0.04), 4
            );
          } else {
            this._rMatrix[stageIdx][prev] = roundTo(
              this._rMatrix[prev][prev] * 0.4, 4
            );
          }
        }

        // Above diagonal: zero-shot transfer
        for (let future = stageIdx + 1; future < SIMULATION_TOTAL_STAGES; future++) {
          this._rMatrix[stageIdx][future] = roundTo(randomInRange(0.20, 0.35), 4);
        }

        // Advance stage
        this._currentStep = 0;
        this._currentStage++;
        if (this._currentStage >= SIMULATION_TOTAL_STAGES) {
          this._isTraining = false;
          this._currentStage = 0;
        }
      }

      // Calculate metrics
      const activeStage = Math.max(0, stageIdx);
      const row = this._rMatrix[activeStage];
      const acc = roundTo((row.reduce((s, v) => s + v, 0) / row.length) * 100, 2);

      let bwt = 0.0;
      if (stageIdx >= 2) {
        let sum = 0;
        for (let j = 0; j < stageIdx; j++) {
          sum += this._rMatrix[stageIdx][j] - this._rMatrix[j][j];
        }
        bwt = roundTo((sum / stageIdx) * 100, 2);
      }

      const learningRate = 3e-4 * (1.0 - progress * 0.5);

      /** @type {import('./types.js').TelemetryPayload} */
      const telemetry = {
        type: 'telemetry_update',
        is_training: this._isTraining,
        stage: stageIdx + 1,
        total_stages: SIMULATION_TOTAL_STAGES,
        task_name: task.key,
        task_label: task.label,
        step: this._currentStep,
        total_steps: SIMULATION_STEPS_PER_STAGE,
        loss: currentLoss,
        vram_mb: vram,
        tokens_per_sec: tokensPerSec,
        learning_rate: learningRate,
        r_matrix: this._rMatrix.map((r) => [...r]),
        acc,
        bwt,
        timestamp: Date.now() / 1000,
      };

      callback(telemetry);
    }, SIMULATION_TICK_INTERVAL_MS);

    return () => {
      this._clearTelemetryInterval();
    };
  }

  // -------------------------------------------------------------------------
  // Inference
  // -------------------------------------------------------------------------

  async runInference(request) {
    await simulatedDelay(150);
    const textLower = (request.text || '').toLowerCase();

    if (request.task_name === 'task_sentiment') {
      const posWords = ['incredible', 'brilliant', 'great', 'masterpiece', 'wonderful', 'breakthrough', 'superb', 'good', 'love'];
      const negWords = ['terrible', 'horrible', 'boring', 'disappointed', 'bad', 'poor', 'waste', 'awful'];

      const posScore = posWords.filter((w) => textLower.includes(w)).length;
      const negScore = negWords.filter((w) => textLower.includes(w)).length;

      let probPos;
      if (posScore > negScore) {
        probPos = roundTo(Math.min(0.98, 0.75 + posScore * 0.08), 4);
      } else if (negScore > posScore) {
        probPos = roundTo(Math.max(0.02, 0.25 - negScore * 0.08), 4);
      } else {
        probPos = 0.52;
      }
      const probNeg = roundTo(1.0 - probPos, 4);

      return {
        task_name: 'task_sentiment',
        task_type: 'sentence_classification',
        prediction: probPos >= 0.5 ? 'Positive' : 'Negative',
        confidence: Math.max(probPos, probNeg),
        probabilities: [
          { class_name: 'Negative', prob: probNeg },
          { class_name: 'Positive', prob: probPos },
        ],
      };
    }

    if (request.task_name === 'task_topic') {
      let topics = [
        { class_name: 'World', prob: 0.12 },
        { class_name: 'Sports', prob: 0.08 },
        { class_name: 'Business', prob: 0.15 },
        { class_name: 'Sci/Tech', prob: 0.65 },
      ];

      if (['summit', 'treaty', 'president', 'united nations'].some((w) => textLower.includes(w))) {
        topics = [{ class_name: 'World', prob: 0.85 }, { class_name: 'Sports', prob: 0.05 }, { class_name: 'Business', prob: 0.06 }, { class_name: 'Sci/Tech', prob: 0.04 }];
      } else if (['championship', 'game', 'team', 'goal', 'cup', 'overtime'].some((w) => textLower.includes(w))) {
        topics = [{ class_name: 'World', prob: 0.04 }, { class_name: 'Sports', prob: 0.88 }, { class_name: 'Business', prob: 0.05 }, { class_name: 'Sci/Tech', prob: 0.03 }];
      } else if (['market', 'bank', 'stock', 'revenue', 'profit', 'inflation'].some((w) => textLower.includes(w))) {
        topics = [{ class_name: 'World', prob: 0.06 }, { class_name: 'Sports', prob: 0.03 }, { class_name: 'Business', prob: 0.82 }, { class_name: 'Sci/Tech', prob: 0.09 }];
      }

      const topPred = topics.reduce((max, t) => (t.prob > max.prob ? t : max), topics[0]);
      return {
        task_name: 'task_topic',
        task_type: 'sentence_classification',
        prediction: topPred.class_name,
        confidence: topPred.prob,
        probabilities: topics,
      };
    }

    if (request.task_name === 'task_ner') {
      const words = (request.text || '').trim().split(/\s+/);
      const knownPer = ['Barack', 'Obama', 'Jamal', 'Al-Mansoub', 'Elon', 'Musk'];
      const knownOrg = ['Google', 'Apple', 'Microsoft', 'NASA'];
      const knownLoc = ['Geneva', 'Cairo', 'London', 'Europe', 'Paris'];

      const tokens = words.map((w) => {
        const clean = w.replace(/[,.!?]/g, '');
        let tag = 'O';
        if (knownPer.includes(clean)) tag = 'PER';
        else if (knownOrg.includes(clean)) tag = 'ORG';
        else if (knownLoc.includes(clean)) tag = 'LOC';
        return { word: w, tag, is_entity: tag !== 'O' };
      });

      return {
        task_name: 'task_ner',
        task_type: 'token_classification',
        tokens,
      };
    }

    return {
      task_name: request.task_name,
      prediction: 'Success',
      text: request.text,
    };
  }

  // -------------------------------------------------------------------------
  // Educational Simulations
  // -------------------------------------------------------------------------

  async getRopeSimulation(params = {}) {
    await simulatedDelay();
    const tokenIndex = params.token_index ?? 1;
    const dim = params.dim ?? 16;
    const base = params.base ?? 10000.0;
    const numPairs = Math.floor(dim / 2);
    const pairs = [];

    for (let i = 0; i < numPairs; i++) {
      const thetaI = 1.0 / Math.pow(base, (2 * i) / dim);
      const angleRad = tokenIndex * thetaI;
      const angleDeg = ((angleRad * 180) / Math.PI) % 360;

      const baseX = 1.0;
      const baseY = 0.0;
      const rotX = baseX * Math.cos(angleRad) - baseY * Math.sin(angleRad);
      const rotY = baseX * Math.sin(angleRad) + baseY * Math.cos(angleRad);

      pairs.push({
        pair_index: i,
        dim_indices: [2 * i, 2 * i + 1],
        theta_i: roundTo(thetaI, 6),
        angle_rad: roundTo(angleRad, 4),
        angle_deg: roundTo(angleDeg, 2),
        vector_original: [roundTo(baseX, 4), roundTo(baseY, 4)],
        vector_rotated: [roundTo(rotX, 4), roundTo(rotY, 4)],
        norm_preserved: roundTo(Math.sqrt(rotX * rotX + rotY * rotY), 4),
      });
    }

    return {
      token_index: tokenIndex,
      total_dim: dim,
      num_pairs: numPairs,
      base_theta: base,
      pairs,
    };
  }

  async getFlashAttention(params = {}) {
    await simulatedDelay();
    const seqLen = params.seq_len ?? 1024;
    const hiddenDim = params.hidden_dim ?? 512;
    const numHeads = params.num_heads ?? 8;
    const blockSize = params.block_size ?? 64;
    const headDim = Math.floor(hiddenDim / numHeads);
    const bytesPerElem = 2;

    const standardMatrixBytes = seqLen * seqLen * bytesPerElem * numHeads;
    const standardHbmRwBytes =
      (seqLen * hiddenDim * 3 + seqLen * seqLen * numHeads + seqLen * hiddenDim) * bytesPerElem;

    const numBlocks = Math.ceil(seqLen / blockSize);
    const flashSramBlockBytes = blockSize * headDim * 3 * bytesPerElem * numHeads;
    const flashHbmRwBytes =
      (seqLen * hiddenDim * 3 + seqLen * hiddenDim) * bytesPerElem;

    const hbmSavings = roundTo(
      Math.max(0, (1.0 - flashHbmRwBytes / Math.max(1, standardHbmRwBytes)) * 100.0), 2
    );

    return {
      seq_len: seqLen,
      hidden_dim: hiddenDim,
      num_heads: numHeads,
      head_dim: headDim,
      block_size: blockSize,
      num_blocks_per_seq: numBlocks,
      standard_attention: {
        matrix_footprint_mb: roundTo(standardMatrixBytes / (1024 * 1024), 3),
        hbm_transfers_mb: roundTo(standardHbmRwBytes / (1024 * 1024), 3),
        complexity: `O(N^2) = ${(seqLen * seqLen).toLocaleString()} elements`,
      },
      flash_attention: {
        sram_footprint_kb: roundTo(flashSramBlockBytes / 1024, 2),
        hbm_transfers_mb: roundTo(flashHbmRwBytes / (1024 * 1024), 3),
        complexity: `O(N) = ${(seqLen * blockSize).toLocaleString()} block computations`,
      },
      hbm_memory_traffic_reduction_percent: hbmSavings,
      estimated_speedup_factor: roundTo(Math.max(1.8, 4.2 * (seqLen / 512)), 1),
    };
  }

  async getContinualLearning(params = {}) {
    await simulatedDelay();
    const useEwc = params.use_ewc !== false && params.use_si !== false;
    const ewcLambda = params.ewc_lambda ?? params.si_c ?? 1000.0;
    const numSteps = params.num_steps ?? 30;
    const half = Math.floor(numSteps / 2);

    const steps = [];
    const task1Acc = [];
    const task2Acc = [];
    const fisherVals = [];

    for (let t = 1; t <= numSteps; t++) {
      steps.push(t);
      let t1, t2, fisher;

      if (t <= half) {
        const p = t / half;
        t1 = 0.5 + 0.45 * (1.0 - Math.exp(-3 * p));
        t2 = 0.25 + 0.05 * Math.sin(p);
        fisher = useEwc ? 0.8 * p : 0.0;
      } else {
        const p = (t - half) / half;
        t2 = 0.25 + 0.65 * (1.0 - Math.exp(-3 * p));
        if (useEwc) {
          const decayResist = Math.min(1.0, 0.95 - 0.05 / (ewcLambda / 500 + 0.1));
          t1 = 0.95 * (1.0 - (1.0 - decayResist) * p);
          fisher = 0.8 + 0.7 * p;
        } else {
          t1 = 0.95 * Math.exp(-4.5 * p);
          fisher = 0.0;
        }
      }

      task1Acc.push(roundTo(t1 * 100, 2));
      task2Acc.push(roundTo(t2 * 100, 2));
      fisherVals.push(roundTo(fisher, 3));
    }

    const bwt = roundTo(task1Acc[task1Acc.length - 1] - task1Acc[half - 1], 2);
    const acc = roundTo((task1Acc[task1Acc.length - 1] + task2Acc[task2Acc.length - 1]) / 2.0, 2);

    return {
      use_ewc: useEwc,
      ewc_lambda: ewcLambda,
      steps,
      task1_accuracy: task1Acc,
      task2_accuracy: task2Acc,
      fisher_importance: fisherVals,
      final_metrics: {
        task1_final: task1Acc[task1Acc.length - 1],
        task2_final: task2Acc[task2Acc.length - 1],
        bwt,
        average_acc: acc,
        forgetting_prevented: bwt > -15.0,
      },
    };
  }

  async getAdaLoRASVD(params = {}) {
    await simulatedDelay();
    const initRank = params.init_rank ?? 16;
    const targetRank = params.target_rank ?? 4;
    const dIn = params.d_in ?? 128;
    const dOut = params.d_out ?? 128;

    const singularValues = [];
    const sensitivities = [];
    for (let i = 0; i < initRank; i++) {
      const sv = roundTo(1.0 / (1.0 + 0.4 * i), 4);
      singularValues.push(sv);
      sensitivities.push(roundTo(sv * (0.8 + 0.4 * Math.sin(i)), 4));
    }

    const sortedIndices = Array.from({ length: initRank }, (_, i) => i)
      .sort((a, b) => sensitivities[b] - sensitivities[a]);
    const retainedSet = new Set(sortedIndices.slice(0, targetRank));

    const denseParams = dIn * dOut;
    const initLoraParams = initRank * (dIn + dOut) + initRank;
    const prunedLoraParams = targetRank * (dIn + dOut) + targetRank;

    const ranks = [];
    for (let i = 0; i < initRank; i++) {
      ranks.push({
        rank_index: i + 1,
        singular_value: singularValues[i],
        importance_score: sensitivities[i],
        status: retainedSet.has(i) ? 'retained' : 'pruned',
      });
    }

    return {
      d_in: dIn,
      d_out: dOut,
      init_rank: initRank,
      target_rank: targetRank,
      dense_params: denseParams,
      init_lora_params: initLoraParams,
      pruned_lora_params: prunedLoraParams,
      param_reduction_vs_dense_percent: roundTo((1.0 - prunedLoraParams / denseParams) * 100, 2),
      param_reduction_vs_init_lora_percent: roundTo((1.0 - prunedLoraParams / initLoraParams) * 100, 2),
      ranks,
    };
  }

  async tokenizeAlign(request = {}) {
    await simulatedDelay();
    const text = request.text || 'Barack Obama visited the United Nations in New York';
    const taskType = request.task_type || 'token_classification';
    const words = text.trim().split(/\s+/);

    const mockSubwords = {
      Afghanistan: ['Af', '##ghan', '##istan'],
      Barack: ['Bar', '##ack'],
      Obama: ['Obama'],
      visited: ['visited'],
      the: ['the'],
      United: ['United'],
      Nations: ['Nations'],
      headquarters: ['head', '##quarters'],
      in: ['in'],
      New: ['New'],
      York: ['York'],
      yesterday: ['yesterday'],
    };

    const mockTags = {
      Barack: 'B-PER', Obama: 'I-PER',
      United: 'B-ORG', Nations: 'I-ORG',
      New: 'B-LOC', York: 'I-LOC',
    };

    const tokens = [];
    let globalId = 0;

    for (const word of words) {
      const subPieces = mockSubwords[word] ||
        (word.length <= 5 ? [word] : [word.slice(0, 4), '##' + word.slice(4)]);
      const wordTag = mockTags[word] || 'O';

      for (let idx = 0; idx < subPieces.length; idx++) {
        const isFirst = idx === 0;
        let effectiveLabel, lossBehavior;

        if (taskType === 'token_classification') {
          effectiveLabel = isFirst ? wordTag : '-100';
          lossBehavior = isFirst ? 'active (computed in loss)' : 'ignored (-100 mask)';
        } else {
          effectiveLabel = 'Sequence Classification (Global)';
          lossBehavior = 'aggregated';
        }

        tokens.push({
          token_id: globalId,
          token_piece: subPieces[idx],
          parent_word: word,
          is_subword_continuation: !isFirst,
          assigned_label: effectiveLabel,
          loss_behavior: lossBehavior,
        });
        globalId++;
      }
    }

    return {
      original_text: text,
      word_count: words.length,
      total_tokens: tokens.length,
      task_type: taskType,
      tokens,
    };
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  /** @private */
  _getTaskKey(stageIndex) {
    const keys = ['task_sentiment', 'task_topic', 'task_ner'];
    return keys[stageIndex] || keys[0];
  }

  /** @private */
  _clearTelemetryInterval() {
    if (this._telemetryInterval !== null) {
      clearInterval(this._telemetryInterval);
      this._telemetryInterval = null;
    }
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  dispose() {
    this._isTraining = false;
    this._clearTelemetryInterval();
  }
}
