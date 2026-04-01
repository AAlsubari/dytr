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
Transformer decoder implementation for Dynamic Transformers.

This module contains the decoder layers and the complete transformer decoder
with support for cross-attention, causal masking, and text generation.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dytr.core.layers import FeedForward, MultiHeadAttention


class DecoderLayer(nn.Module):
    """
    Single transformer decoder layer with self-attention, cross-attention, and feedforward network.

    Args:
        layer_id: Layer index
        embed_dim: Embedding dimension
        config: Model configuration
        has_cross_attention: Whether to include cross-attention (for seq2seq)
    """

    def __init__(self, layer_id: int, embed_dim: int, config, has_cross_attention: bool = True):
        super().__init__()
        self.layer_id = layer_id
        self.embed_dim = embed_dim
        self.config = config
        self.has_cross_attention = has_cross_attention

        # Self-attention (causal)
        self.self_attention = MultiHeadAttention(
            embed_dim, config.num_heads, config.head_dim, config, layer_id, is_cross_attention=False
        )

        # Cross-attention (if enabled)
        if has_cross_attention:
            self.cross_attention = MultiHeadAttention(
                embed_dim,
                config.num_heads,
                config.head_dim,
                config,
                layer_id,
                is_cross_attention=True,
            )

        # Feedforward network
        self.ffn = FeedForward(embed_dim, config.ff_mult, config, layer_id)

        # Layer normalizations
        self.norm1 = nn.LayerNorm(embed_dim)
        if has_cross_attention:
            self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Task adapters
        if config.use_task_adapters:
            self.self_attn_adapter = nn.Sequential(
                nn.Linear(embed_dim, config.adapter_bottleneck),
                nn.GELU(),
                nn.Linear(config.adapter_bottleneck, embed_dim),
            )
            if has_cross_attention:
                self.cross_attn_adapter = nn.Sequential(
                    nn.Linear(embed_dim, config.adapter_bottleneck),
                    nn.GELU(),
                    nn.Linear(config.adapter_bottleneck, embed_dim),
                )
            self.ffn_adapter = nn.Sequential(
                nn.Linear(embed_dim, config.adapter_bottleneck),
                nn.GELU(),
                nn.Linear(config.adapter_bottleneck, embed_dim),
            )
        else:
            self.self_attn_adapter = None
            self.cross_attn_adapter = None
            self.ffn_adapter = None

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        task_name: Optional[str] = None,
    ):
        """
        Forward pass through decoder layer.

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            encoder_output: Encoder output for cross-attention
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)
            task_name: Current task name for adapter selection
        """
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, encoder_output=None, mask=tgt_mask, is_causal=True)
        if self.self_attn_adapter is not None and task_name is not None:
            x = x + self.self_attn_adapter(x)
        x = residual + x

        # Cross-attention with residual connection (if enabled)
        if self.has_cross_attention and encoder_output is not None:
            residual = x
            x = self.norm2(x)
            x = self.cross_attention(
                x, encoder_output=encoder_output, mask=src_mask, is_causal=False
            )
            if self.cross_attn_adapter is not None and task_name is not None:
                x = x + self.cross_attn_adapter(x)
            x = residual + x

        # Feedforward with residual connection
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        if self.ffn_adapter is not None and task_name is not None:
            x = x + self.ffn_adapter(x)
        x = residual + x

        return x


class TransformerDecoder(nn.Module):
    """
    Complete transformer decoder consisting of multiple decoder layers.

    Supports both causal LM (no cross-attention) and seq2seq (with cross-attention) modes.

    Args:
        config: Model configuration
        task_name: Name of the task this decoder belongs to
        has_cross_attention: Whether to include cross-attention layers
    """

    def __init__(self, config, task_name: str, has_cross_attention: bool = True):
        super().__init__()
        self.config = config
        self.task_name = task_name
        self.has_cross_attention = has_cross_attention

        # Embedding will be set later to share with encoder
        self.embedding = None

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(i, config.embed_dim, config, has_cross_attention)
                for i in range(config.num_layers)
            ]
        )

        # Final layer norm and output projection
        self.norm = nn.LayerNorm(config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size)

    def set_shared_embedding(self, embedding: nn.Embedding):
        """Share embedding weights with encoder."""
        self.embedding = embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        task_name: Optional[str] = None,
    ):
        """
        Forward pass through transformer decoder.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            encoder_output: Encoder output for cross-attention
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            task_name: Current task name for adapter selection
        """
        x = self.embedding(input_ids)

        for layer in self.layers:
            if self.has_cross_attention:
                x = layer(x, encoder_output, src_mask, tgt_mask, task_name or self.task_name)
            else:
                x = layer(x, None, None, tgt_mask, task_name or self.task_name)

        x = self.norm(x)
        logits = self.output_proj(x)
        return logits

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ):
        """
        Generate text using the decoder.

        Args:
            input_ids: Optional input token IDs for causal LM
            encoder_output: Encoder output for seq2seq
            src_mask: Source attention mask
            max_len: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 to disable)
            top_p: Top-p (nucleus) sampling (0 to disable)
            repetition_penalty: Penalty for repeating tokens
        """
        batch_size = encoder_output.size(0) if encoder_output is not None else input_ids.size(0)
        device = encoder_output.device if encoder_output is not None else input_ids.device

        bos_id = self.config.bos_token_id or 101  # Default BOS token
        eos_id = self.config.eos_token_id or 102  # Default EOS token

        # Initialize generated sequence
        if input_ids is not None:
            generated = input_ids.clone()
        else:
            generated = torch.ones(batch_size, 1, dtype=torch.long, device=device) * bos_id

        # Generate tokens autoregressively
        for _ in range(max_len):
            # Create causal mask for current sequence
            tgt_mask = torch.ones(batch_size, generated.size(1), device=device)

            # Forward pass
            if self.has_cross_attention:
                logits = self.forward(generated, encoder_output, src_mask, tgt_mask, self.task_name)
            else:
                logits = self.forward(generated, tgt_mask=tgt_mask, task_name=self.task_name)

            # Get next token logits and apply temperature
            next_token_logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            for token_id in set(generated[0].tolist()):
                next_token_logits[:, token_id] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for batch_idx in range(next_token_logits.size(0)):
                    indices_to_remove = sorted_indices[batch_idx][
                        sorted_indices_to_remove[batch_idx]
                    ]
                    next_token_logits[batch_idx, indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if all sequences have generated EOS
            if (next_token == eos_id).all():
                break

        return generated
