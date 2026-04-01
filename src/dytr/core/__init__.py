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
Core transformer components for Dynamic Transformers.

This module contains the main model classes, configuration, and core building blocks
for the transformer architecture including attention mechanisms, feedforward networks,
and the main DynamicTransformer class.
"""

from dytr.core.config import ModelConfig, TaskConfig, TrainingStrategy
from dytr.core.decoder import DecoderLayer, TransformerDecoder
from dytr.core.encoder import EncoderLayer, TransformerEncoder
from dytr.core.exporter import ModelExporter
from dytr.core.layers import (
    FeedForward,
    MultiHeadAttention,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
from dytr.core.model import DynamicTransformer

__all__ = [
    "DynamicTransformer",
    "ModelConfig",
    "TaskConfig",
    "TrainingStrategy",
    "RotaryEmbedding",
    "MultiHeadAttention",
    "FeedForward",
    "rotate_half",
    "apply_rotary_pos_emb",
    "TransformerEncoder",
    "EncoderLayer",
    "TransformerDecoder",
    "DecoderLayer",
    "ModelExporter",
]

