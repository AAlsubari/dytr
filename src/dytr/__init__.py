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
dytr - Dynamic Transformer for Multi-Task Learning with Continual Learning Support

A flexible PyTorch library for multi-task learning with transformer architectures.


"""

__version__ = "0.1.0"
__author__ = "Akram Alsubari"
__email__ = "akram.alsubari@outlook.com"
__license__ = "Apache 2.0"

from dytr.core.config import ModelConfig, TaskConfig, TrainingStrategy
from dytr.core.exporter import ModelExporter

# Core model components
from dytr.core.model import DynamicTransformer

# Memory/Continual learning components
from dytr.memory.ewc import EWC
from dytr.memory.replay import ReplayBuffer
from dytr.tokenization.download_manager import (
    DownloadManager,
    download_file,
    download_tokenizer_files,
    get_url_from_HF,
    process_vocab_text,
)

# Tokenizer
from dytr.tokenization.simple_tokenizer import SimpleTokenizer
# Pretrained model loading
#from dytr.pretrained import PretrainedModelLoader
from dytr.pretrained.loader import PretrainedModelLoader
from dytr.training.data import MultiTaskDataset, TaskAwareBatchSampler, collate_fn
from dytr.training.dataset import SingleDatasetProcessing

# Training components
from dytr.training.trainer import Trainer

# Utility functions
from dytr.training.utils import process_qa_dataset, set_seed
from dytr.utils.caching import CacheManager, get_cache_manager
from dytr.utils.logging import get_logger, set_log_level, disable_logging


# Define what gets imported with "from dynamic_transformers import *"
__all__ = [
    # Core classes
    "DynamicTransformer",
    "ModelConfig",
    "TaskConfig",
    "TrainingStrategy",
    "Trainer",
    "ModelExporter",
    # Dataset classes
    "SingleDatasetProcessing",
    "MultiTaskDataset",
    "TaskAwareBatchSampler",
    "collate_fn",
    # Continual learning
    "EWC",
    "ReplayBuffer",
    # Pretrained Models
    "PretrainedModelLoader",
    # Tokenizer
    "SimpleTokenizer",
    "DownloadManager",
    "download_file",
    "download_tokenizer_files",
    "get_url_from_HF",
    "process_vocab_text",
    # Utilities
    "set_seed",
    "process_qa_dataset",
    "CacheManager",
    "get_cache_manager",
    "get_logger",
    "set_log_level", 
    "disable_logging",
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

