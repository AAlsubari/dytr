// ---------------------------------------------------------------------------
// DyTrStudio Communication Layer - Unified Data Schemas (JSDoc Typed)
// ---------------------------------------------------------------------------
// Author: Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// Licensed under the Apache License, Version 2.0
// ---------------------------------------------------------------------------
//
// All type definitions are shared between FastApiClient and SimulationClient.
// Both clients MUST return objects conforming to these exact shapes.
// React components consume these types without awareness of the data source.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Health & System
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} HealthCheckResponse
 * @property {string}  status         - Server status: 'online' | 'offline'
 * @property {string}  service        - Service identifier
 * @property {string}  version        - Backend version string
 * @property {boolean} dytr_available - Whether the dytr library is loaded
 * @property {string}  engine_mode    - Resolved mode: 'live' | 'simulation'
 */

/**
 * @typedef {Object} SystemCapabilities
 * @property {string}  dytr_version              - DyTR library version
 * @property {boolean} dytr_available             - Library availability flag
 * @property {boolean} cuda_available              - GPU compute available
 * @property {string}  device_name                 - GPU/CPU device identifier
 * @property {number}  vram_total_gb               - Total VRAM in GB
 * @property {boolean} flash_attention_supported   - FlashAttention SDPA support
 * @property {boolean} amp_supported               - Automatic Mixed Precision
 * @property {boolean} bf16_supported              - BFloat16 precision support
 * @property {string}  torch_version               - PyTorch version
 */

// ---------------------------------------------------------------------------
// Model Configuration & Estimation
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} TaskDefinition
 * @property {string}  task_name   - Internal task key (e.g., 'task_sentiment')
 * @property {string}  task_type   - Classification strategy identifier
 * @property {number}  num_classes - Number of output classes
 * @property {number}  max_len     - Maximum sequence length for this task
 */

/**
 * @typedef {Object} ModelPreset
 * @property {string}         id          - Unique preset identifier
 * @property {string}         name        - Human-readable preset name
 * @property {string}         description - What this preset does
 * @property {string}         category    - Preset category grouping
 * @property {string}         badge       - UI badge label
 * @property {ModelBuildConfig} config    - Architecture configuration
 */

/**
 * @typedef {Object} ModelBuildConfig
 * @property {number}           vocab_size             - Vocabulary size
 * @property {number}           embed_dim              - Embedding dimension
 * @property {number}           num_layers             - Transformer encoder layers
 * @property {number}           num_heads              - Attention heads
 * @property {number}           [head_dim]             - Per-head dimension
 * @property {number}           ff_mult                - FFN expansion multiplier
 * @property {number}           max_seq_len            - Max sequence length
 * @property {number}           dropout                - Dropout rate
 * @property {boolean}          use_flash_attention    - FlashAttention toggle
 * @property {boolean}          use_rotary_embedding   - RoPE toggle
 * @property {boolean}          [gradient_checkpointing] - Memory optimization
 * @property {boolean}          use_amp                - Mixed precision toggle
 * @property {boolean}          use_ewc                - Elastic Weight Consolidation
 * @property {number}           ewc_lambda             - EWC regularization strength
 * @property {TaskDefinition[]} tasks                  - Multi-task heads
 */

/**
 * @typedef {Object} HeadBreakdown
 * @property {string} task_name - Task identifier
 * @property {string} task_type - Classification type
 * @property {number} params    - Parameter count for this head
 */

/**
 * @typedef {Object} ModelEstimate
 * @property {number}          total_parameters           - Total parameter count
 * @property {string}          total_parameters_formatted - Formatted with commas
 * @property {number}          embedding_params           - Embedding layer params
 * @property {number}          encoder_params             - Encoder stack params
 * @property {number}          head_params_total          - All task heads combined
 * @property {HeadBreakdown[]} head_breakdowns            - Per-head breakdown
 * @property {{ fp32_mb: number, fp16_amp_mb: number }} memory_footprint
 * @property {string}          python_code                - Generated Python script
 */

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} DatasetSampleRow
 * @property {string}          text              - Sample text content
 * @property {number}          [label]           - Numeric label (classification)
 * @property {string}          [tags]            - Comma-separated NER tags
 * @property {string}          [annotated_preview] - Human-readable entity annotations
 */

/**
 * @typedef {Object} Dataset
 * @property {string}             id          - Unique dataset identifier
 * @property {string}             name        - Display name
 * @property {string}             task_name   - Associated task key
 * @property {string}             task_type   - Classification strategy
 * @property {number}             num_samples - Total sample count
 * @property {string[]}           classes     - Class label descriptions
 * @property {string}             description - Dataset description
 * @property {DatasetSampleRow[]} sample_rows - Preview rows
 */

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} TrainingConfig
 * @property {number}  num_epochs_per_task   - Epochs per task stage
 * @property {number}  learning_rate         - Initial learning rate
 * @property {number}  batch_size            - Training batch size
 * @property {boolean} use_flash_attention   - FlashAttention during training
 * @property {boolean} use_amp              - Mixed precision during training
 * @property {boolean} use_ewc              - Enable Elastic Weight Consolidation
 * @property {number}  ewc_lambda           - EWC regularization penalty
 * @property {boolean} [use_replay]         - Experience replay buffer toggle
 */

/**
 * @typedef {Object} TrainingState
 * @property {boolean}    is_training    - Whether a session is active
 * @property {number}     current_stage  - Current task stage index
 * @property {number}     total_stages   - Total task stages
 * @property {string}     current_task   - Active task key
 * @property {number}     step           - Current step within stage
 * @property {number}     total_steps    - Total steps per stage
 * @property {number}     loss           - Current loss value
 * @property {number}     vram_mb        - Current VRAM usage in MB
 * @property {number}     tokens_per_sec - Throughput
 * @property {number[][]} r_matrix       - 3x3 R-Matrix
 * @property {{ acc: number, bwt: number, fwt: number }} metrics
 */

/**
 * @typedef {Object} TelemetryPayload
 * @property {string}     type           - Always 'telemetry_update'
 * @property {boolean}    is_training    - Session active flag
 * @property {number}     stage          - Current stage (1-indexed)
 * @property {number}     total_stages   - Total stages
 * @property {string}     task_name      - Task identifier
 * @property {string}     task_label     - Human-readable task name
 * @property {number}     step           - Step within current stage
 * @property {number}     total_steps    - Total steps in stage
 * @property {number}     loss           - Current loss
 * @property {number}     vram_mb        - VRAM consumption in MB
 * @property {number}     tokens_per_sec - Processing throughput
 * @property {number}     learning_rate  - Current LR (with decay)
 * @property {number[][]} r_matrix       - 3x3 recall/transfer matrix
 * @property {number}     acc            - Average accuracy
 * @property {number}     bwt            - Backward Transfer metric
 * @property {number}     timestamp      - Unix timestamp
 */

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} InferenceRequest
 * @property {string} task_name - Target task head
 * @property {string} text      - Input text for prediction
 */

/**
 * @typedef {Object} ClassProbability
 * @property {string} class_name - Class label
 * @property {number} prob       - Probability score
 */

/**
 * @typedef {Object} TokenEntity
 * @property {string}  word      - Original word
 * @property {string}  tag       - NER tag
 * @property {boolean} is_entity - Whether this token is an entity
 */

/**
 * @typedef {Object} InferenceResult
 * @property {string}              task_name     - Task identifier
 * @property {string}              task_type     - Classification type
 * @property {string}              [prediction]  - Predicted label
 * @property {number}              [confidence]  - Confidence score
 * @property {ClassProbability[]}  [probabilities] - Class-wise probabilities
 * @property {TokenEntity[]}       [tokens]      - Token-level NER results
 */

// ---------------------------------------------------------------------------
// Simulations (Educational Glassroom)
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} RopePair
 * @property {number}   pair_index      - Dimension pair index
 * @property {number[]} dim_indices     - [2i, 2i+1]
 * @property {number}   theta_i         - Angular frequency
 * @property {number}   angle_rad       - Rotation angle in radians
 * @property {number}   angle_deg       - Rotation angle in degrees
 * @property {number[]} vector_original - Base vector [x, y]
 * @property {number[]} vector_rotated  - Rotated vector [x, y]
 * @property {number}   norm_preserved  - L2 norm (should be ~1.0)
 */

/**
 * @typedef {Object} RopeData
 * @property {number}     token_index - Position m
 * @property {number}     total_dim   - Full embedding dimension
 * @property {number}     num_pairs   - Number of 2D pairs
 * @property {number}     base_theta  - Base frequency
 * @property {RopePair[]} pairs       - Per-pair rotation data
 */

/**
 * @typedef {Object} FlashAttentionData
 * @property {number} seq_len
 * @property {number} hidden_dim
 * @property {number} num_heads
 * @property {number} head_dim
 * @property {number} block_size
 * @property {number} num_blocks_per_seq
 * @property {{ matrix_footprint_mb: number, hbm_transfers_mb: number, complexity: string }} standard_attention
 * @property {{ sram_footprint_kb: number, hbm_transfers_mb: number, complexity: string }} flash_attention
 * @property {number} hbm_memory_traffic_reduction_percent
 * @property {number} estimated_speedup_factor
 */

/**
 * @typedef {Object} ContinualLearningData
 * @property {boolean}  use_ewc              - EWC penalty enabled
 * @property {number}   ewc_lambda           - Fisher penalty regularization strength
 * @property {number[]} steps                - Time step indices
 * @property {number[]} task1_accuracy       - Task 1 accuracy trajectory
 * @property {number[]} task2_accuracy       - Task 2 accuracy trajectory
 * @property {number[]} fisher_importance    - Diagonal Fisher values
 * @property {{ task1_final: number, task2_final: number, bwt: number, average_acc: number, forgetting_prevented: boolean }} final_metrics
 */

/**
 * @typedef {Object} AdaLoRARankInfo
 * @property {number} rank_index       - 1-indexed rank position
 * @property {number} singular_value   - Sigma value
 * @property {number} importance_score - Sensitivity score
 * @property {string} status           - 'retained' | 'pruned'
 */

/**
 * @typedef {Object} AdaLoRAData
 * @property {number}            d_in
 * @property {number}            d_out
 * @property {number}            init_rank
 * @property {number}            target_rank
 * @property {number}            dense_params
 * @property {number}            init_lora_params
 * @property {number}            pruned_lora_params
 * @property {number}            param_reduction_vs_dense_percent
 * @property {number}            param_reduction_vs_init_lora_percent
 * @property {AdaLoRARankInfo[]} ranks
 */

/**
 * @typedef {Object} TokenDetail
 * @property {number}  token_id                  - Global token index
 * @property {string}  token_piece               - Subword piece
 * @property {string}  parent_word               - Original word
 * @property {boolean} is_subword_continuation   - Continuation flag
 * @property {string}  assigned_label            - Active label or '-100'
 * @property {string}  loss_behavior             - Loss computation status
 */

/**
 * @typedef {Object} TokenAlignData
 * @property {string}        original_text - Input text
 * @property {number}        word_count    - Number of words
 * @property {number}        total_tokens  - Total subword tokens
 * @property {string}        task_type     - Classification strategy
 * @property {TokenDetail[]} tokens        - Per-token alignment detail
 */

// This module exports nothing at runtime.
// It exists solely for JSDoc type reference across the communication layer.
export {};
