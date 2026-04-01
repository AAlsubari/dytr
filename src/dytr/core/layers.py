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
Core transformer layers including attention mechanisms and feedforward networks.

This module implements the fundamental building blocks of the transformer architecture:
multi-head attention with rotary positional embeddings, feedforward networks,
and supporting utility functions.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformer models.

    Implements rotary positional embeddings that encode position information
    by rotating query and key vectors in the complex plane.

    Args:
        dim: Dimension of the embedding
        max_seq_len: Maximum sequence length
        base: Base for the frequency computation
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build and cache cosine and sine values for rotary embeddings."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int):
        """Get cosine and sine values for rotary embeddings."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism with support for cross-attention and rotary embeddings.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        config: Model configuration
        layer_id: Layer index for debugging
        is_cross_attention: Whether this is cross-attention layer
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        config,
        layer_id: int,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim
        self.scale = head_dim**-0.5
        self.config = config
        self.layer_id = layer_id
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Linear(embed_dim, self.total_dim)
        self.k_proj = nn.Linear(embed_dim, self.total_dim)
        self.v_proj = nn.Linear(embed_dim, self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        if config.use_rotary_embedding and not is_cross_attention:
            self.rotary = RotaryEmbedding(head_dim, config.max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        """
        Forward pass for multi-head attention.

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            encoder_output: Encoder output for cross-attention
            mask: Attention mask
            is_causal: Whether to apply causal masking
        """
        batch_size, seq_len, _ = x.shape

        # Project and reshape queries
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle keys and values (self-attention or cross-attention)
        if self.is_cross_attention and encoder_output is not None:
            k = (
                self.k_proj(encoder_output)
                .view(batch_size, encoder_output.size(1), self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(encoder_output)
                .view(batch_size, encoder_output.size(1), self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            kv_seq_len = encoder_output.size(1)
        else:
            k = (
                self.k_proj(x)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(x)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            kv_seq_len = seq_len

        # Apply rotary embeddings
        if (
            self.config.use_rotary_embedding
            and not self.is_cross_attention
            and encoder_output is None
        ):
            cos, sin = self.rotary(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply masks
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            if mask.size(-1) != kv_seq_len:
                if mask.size(-1) > kv_seq_len:
                    mask = mask[..., :kv_seq_len]
                else:
                    pad_len = kv_seq_len - mask.size(-1)
                    mask = F.pad(mask, (0, pad_len), value=1)
            if mask.size(1) != self.num_heads:
                mask = mask.expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Apply causal mask if needed
        if is_causal and not self.is_cross_attention:
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_seq_len, device=x.device) * float("-inf"), diagonal=1
            )
            attn_scores = attn_scores + causal_mask[None, None, :, :]

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.total_dim)
        output = self.out_proj(context)

        return output


class FeedForward(nn.Module):
    """
    FeedForward network with GELU activation and SwiGLU-style gating.

    Implements a transformer feedforward layer with gating mechanism
    similar to SwiGLU used in modern LLMs.

    Args:
        embed_dim: Embedding dimension
        ff_mult: Multiplier for feedforward dimension
        config: Model configuration
        layer_id: Layer index for debugging
    """

    def __init__(self, embed_dim: int, ff_mult: int, config, layer_id: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = embed_dim * ff_mult
        self.config = config
        self.layer_id = layer_id

        self.gate_proj = nn.Linear(embed_dim, self.ff_dim)
        self.up_proj = nn.Linear(embed_dim, self.ff_dim)
        self.down_proj = nn.Linear(self.ff_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        """Forward pass through feedforward network."""
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))
