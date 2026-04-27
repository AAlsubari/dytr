# Copyright 2025 Akram Alsubari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration classes for Dynamic Transformers.

This module defines the configuration dataclasses for model architecture,
training parameters, and task-specific settings
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TrainingStrategy(Enum):
    """Enumeration of supported training strategies."""

    CAUSAL_LM = "causal_lm"
    SEQ2SEQ = "seq2seq"
    SENTENCE_CLASSIFICATION = "sentence_classification"
    TOKEN_CLASSIFICATION = "token_classification"


@dataclass
class ModelConfig:
    """
    Configuration class for the Dynamic Transformer model.

    Contains all hyperparameters and settings for model architecture,
    training, and continual learning features.
    """

    # Architecture parameters
    embed_dim: int = 256 # Dimension of token embeddings and hidden states
    # Controls model capacity: larger values = more parameters but better representation
    # Common values: 128 (tiny), 256 (small), 512 (base), 768 (large), 1024 (xl)
    # Impact: Quadratically increases total parameters (embed_dim² × layers)
    # Trade-off: Higher = better accuracy but more memory and slower training
    
    num_layers: int = 6 # Number of transformer encoder/decoder layers stacked vertically
    # Each layer adds: Multi-Head Attention + Feed-Forward Network + Layer Norm
    # Common values: 2 (tiny), 4 (small), 6 (base), 12 (large), 24 (xl)
    # Impact: Linearly increases parameters and inference time

    num_heads: int = 8 # Number of parallel attention heads in Multi-Head Attention
    # Must divide embed_dim evenly (embed_dim % num_heads == 0)
    # Common values: 2, 4, 8, 12, 16
    # Each head learns different attention patterns (position, syntax, semantics)
    # Trade-off: More heads = richer representations but more computation


    head_dim: int = 256//8 # embed_dim//num_heads 
    # Dimension of each attention head's query, key, value vectors
    # Calculated as embed_dim / num_heads (typically 64)
    # Controls attention granularity: larger = more detailed relationships
    # set it with caution 



    ff_mult: int = 4 # Multiplier for Feed-Forward Network hidden dimension
    # FFN hidden size = embed_dim × ff_mult
    # Common values: 2, 3, 4, 6, 8
    # Larger values = more capacity for pattern recognition
    # FFN accounts for ~2/3 of total parameters, impacts memory significantly

    dropout: float = 0.1 # Dropout rate for regularization (0.0 = no dropout, 1.0 = all dropped)
    # Applied after attention and FFN layers
    # Common values: 0.0, 0.1, 0.2, 0.3, 0.5
    # Prevents overfitting: higher = more regularization but slower convergence
    # Reduce if underfitting, increase if overfitting

    max_seq_len: int = 256

    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 16
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    max_learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    adam_epsilon: float = 1e-8
    label_smoothing: float = 0.1

    # Advanced training features
    fp16: bool = False ## still need to be implemented 
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Evaluation and checkpointing
    patience: int = 3
    evaluation_strategy: str = "epoch" #When to evaluate: "steps" or "epoch"
    logging_steps: int = 20 
    validation_check_interval: int = 100
    load_best_model_at_end: bool = True # still not integrated 
    metric_for_best_model: str = "loss" # currently only loss is implemented for saving best model due to different metrics to each task strategies 
    early_stopping_patience: int = 10

    # Training duration
    max_train_steps: int = 100000 # skipped
    num_train_epochs: int = 3
    lr_scheduler_type: str = "cosine"

    # Data loading
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    dataloader_num_workers: int = 0 #
    dataloader_pin_memory: bool = True # 

    # Randomness
    seed: int = 42

    # Model enhancements
    task_specific_lr: Dict[str, float] = field(default_factory=dict)
    task_weights: Dict[str, float] = field(default_factory=dict)
    use_rotary_embedding: bool = True
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    training_from_scratch: bool=False # will be remove

    # Special tokens
    special_tokens: Dict[str, str] = field(
        default_factory=lambda: {
            #"task_sep": "<|tasksep|>",
            #"doc_sep": "<|docsep|>",
            #"answer_start": "<|answer|>",
            #"bos": "<s>",
            #"eos": "</s>",
        }
    )

    # Task configuration
    window_size: int = 256
    stride: int = 64
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Tokenizer IDs
    vocab_size: Optional[int] = None
    tokenizer_name: str = "bert-base-multilingual-cased" # it can be local tokenizer folder files, or path to vocab.json 
    add_tab_newline_vocab: bool= False
    use_simple_tokenizer: bool = True # use simple tokenizer without transformers library 
    tokenizer_type: str = 'wordpiece' 
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    # Continual learning
    adapter_bottleneck: int = 64
    use_task_adapters: bool = True
    ewc_lambda: float = 1000.0 # EWC regularization strength. Higher = more protection of old tasks.
    replay_buffer_size: int = 1000 # OPTIONAL - Size of experience replay buffer.
    use_ewc: bool = True # Enable Elastic Weight Consolidation.
    # Note: Computes Fisher matrix after training, applies to future tasks. 
    use_replay: bool = False # need for more test cases

    # Task-specific learning rate multipliers
    causal_lm_window_size: int = 256
    causal_lm_stride: int = 128
    head_lr_mult: float = 1.0 # Learning rate multiplier for task heads
    decoder_lr_mult: float = 1.0  # Learning rate multiplier for task decoders
    shared_lr_mult: float = 0.5 # Learning rate multiplier for shared components
    

    # Device
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"


@dataclass
class TaskConfig:
    """
    Configuration class for individual tasks.

    Defines the dataset, training strategy, and task-specific parameters
    for a single task in multi-task learning.
    """

    # Basic task info
    task_name: str
    training_strategy: TrainingStrategy

    # Dataset configuration
    datasets: List[Dict[str, Any]] = None
    num_labels: Optional[int] = None

    # Data processing
    max_length: int = 256
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    label_to_ids: Optional[Dict[str,int]]= None

    # Column names
    text_column: str = "text"
    label_column: str = "label"
    source_column: str = "source"
    target_column: str = "target"

    # Sampling
    sample_size: Optional[int] = None
    validation_sample_size: int = 100
    test_sample_size: int = 100

    # Task-specific options
    prompt_template: Optional[str] = None
    metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy"])
    shuffle_data: bool = True
    min_text_length: int = 10
    balance_classes: bool = False
    dataset_config: Optional[Dict[str, Any]] = None
    is_generation: bool = False

    # QA specific
    question_column: Optional[str] = None
    answer_column: Optional[str] = None
    conversations_column: Optional[str] = None
