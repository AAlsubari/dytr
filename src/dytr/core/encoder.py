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
Transformer encoder implementation for Dynamic Transformers.

This module contains the encoder layers and the complete transformer encoder
with support for task adapters.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""

from typing import Optional

import torch
import torch.nn as nn

from dytr.core.layers import FeedForward, MultiHeadAttention


class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feedforward network.

    Args:
        layer_id: Layer index
        embed_dim: Embedding dimension
        config: Model configuration
    """

    def __init__(self, layer_id: int, embed_dim: int, config):
        super().__init__()
        self.layer_id = layer_id
        self.embed_dim = embed_dim
        self.config = config

        # Self-attention components
        self.attention = MultiHeadAttention(
            embed_dim, config.num_heads, config.head_dim, config, layer_id, is_cross_attention=False
        )
        self.attention_norm = nn.LayerNorm(embed_dim)

        # Feedforward components
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, config.ff_mult, config, layer_id)

        # Task adapter
        if config.use_task_adapters:
            self.adapter = nn.Sequential(
                nn.Linear(embed_dim, config.adapter_bottleneck),
                nn.GELU(),
                nn.Linear(config.adapter_bottleneck, embed_dim),
            )
        else:
            self.adapter = None

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, task_name: Optional[str] = None
    ):
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Attention mask
            task_name: Current task name for adapter selection
        """
        # Self-attention with residual connection
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, encoder_output=None, mask=mask, is_causal=False)
        x = residual + x

        # Apply task adapter if available
        if self.adapter is not None and task_name is not None:
            x = x + self.adapter(x)

        # Feedforward with residual connection
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TransformerEncoder(nn.Module):
    """
    Complete transformer encoder consisting of multiple encoder layers.

    Args:
        config: Model configuration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(i, config.embed_dim, config) for i in range(config.num_layers)]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        task_name: Optional[str] = None,
    ):
        """
        Forward pass through transformer encoder.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            mask: Attention mask
            task_name: Current task name for adapter selection
        """
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, mask, task_name)

        x = self.norm(x)
        return x
