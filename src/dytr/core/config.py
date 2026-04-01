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
training parameters, and task-specific settings.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
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
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 64
    ff_mult: int = 4
    dropout: float = 0.1
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
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Evaluation and checkpointing
    patience: int = 3
    evaluation_strategy: str = "steps"
    logging_steps: int = 50
    validation_check_interval: int = 500
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    early_stopping_patience: int = 10

    # Training duration
    max_train_steps: int = 100000
    num_train_epochs: int = 3
    lr_scheduler_type: str = "cosine"

    # Data loading
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True

    # Randomness
    seed: int = 42

    # Model enhancements
    task_specific_lr: Dict[str, float] = field(default_factory=dict)
    task_weights: Dict[str, float] = field(default_factory=dict)
    use_rotary_embedding: bool = True
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    training_from_scratch: bool=False

    # Special tokens
    special_tokens: Dict[str, str] = field(
        default_factory=lambda: {
            "task_sep": "<|tasksep|>",
            "doc_sep": "<|docsep|>",
            "answer_start": "<|answer|>",
            "bos": "<s>",
            "eos": "</s>",
        }
    )

    # Task configuration
    window_size: int = 256
    stride: int = 64
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Tokenizer IDs
    vocab_size: Optional[int] = None
    tokenizer_name: str = "bert-base-multilingual-cased"
    add_tab_newline_vocab: bool= False
    use_simple_tokenizer: bool = True
    tokenizer_type: str = 'wordpiece'
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    # Continual learning
    adapter_bottleneck: int = 64
    use_task_adapters: bool = True
    ewc_lambda: float = 1000.0
    replay_buffer_size: int = 1000
    use_ewc: bool = True
    use_replay: bool = True

    # Task-specific learning rate multipliers
    causal_lm_window_size: int = 256
    causal_lm_stride: int = 128
    head_lr_mult: float = 1.0 # Learning rate multiplier for task heads
    decoder_lr_mult: float = 1.0  # Learning rate multiplier for task decoders
    shared_lr_mult: float = 0.1 # Learning rate multiplier for shared components
    

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
