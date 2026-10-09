# Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
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
Model Builder & Architecture Configuration Router for DyTrStudio.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/models", tags=["models"])


class TaskDefinition(BaseModel):
    task_name: Optional[str] = Field("task_default", example="task_sentiment")
    task_type: Optional[str] = Field("sentence_classification", example="sentence_classification")  # sentence_classification, token_classification, causal_lm, seq2seq
    strategy: Optional[str] = None
    num_classes: Optional[int] = Field(2, example=2)
    max_len: Optional[int] = Field(128, example=64)
    classes: Optional[List[Any]] = None

    class Config:
        extra = "allow"


class ModelBuildRequest(BaseModel):
    # Backbone Dimensions
    vocab_size: Optional[int] = Field(30522, ge=100)
    embed_dim: Optional[int] = Field(256, ge=32, le=2048)
    num_layers: Optional[int] = Field(4, ge=1, le=32)
    num_heads: Optional[int] = Field(4, ge=1, le=64)
    head_dim: Optional[int] = Field(64)
    ff_mult: Optional[int] = Field(4)
    max_seq_len: Optional[int] = Field(256)
    dropout: Optional[float] = Field(0.1)

    # Hardware Optimizations
    use_flash_attention: Optional[bool] = Field(True)
    use_rotary_embedding: Optional[bool] = Field(True)
    gradient_checkpointing: Optional[bool] = Field(False)
    use_amp: Optional[bool] = Field(True)

    # Continual Learning & PEFT
    use_si: Optional[bool] = Field(True)
    si_c: Optional[float] = Field(1.0)
    use_adalora: Optional[bool] = Field(False)
    adalora_target_rank: Optional[int] = Field(4)
    adalora_init_rank: Optional[int] = Field(12)

    # Multi-Task Setup
    tasks: Optional[List[Any]] = Field(default_factory=list)

    class Config:
        extra = "allow"


@router.get("/presets")
async def get_model_presets():
    """Return pre-configured architecture archetypes for easy one-click selection."""
    return [
        {
            "id": "sentiment_analyst",
            "name": "Smart Sentiment Analyst",
            "description": "Lightweight sequence classification model optimized for emotional tone and polarity.",
            "category": "Classification",
            "badge": "Fast & Efficient",
            "config": {
                "embed_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "head_dim": 32,
                "max_seq_len": 128,
                "use_flash_attention": True,
                "use_rotary_embedding": True,
                "use_amp": True,
                "use_si": True,
                "tasks": [
                    {
                        "task_name": "task_sentiment",
                        "task_type": "sentence_classification",
                        "num_classes": 2,
                        "max_len": 64,
                    }
                ],
            },
        },
        {
            "id": "multi_task_champion",
            "name": "Multi-Task Continual Champion",
            "description": "Full Continual Learning benchmark model with Sentiment, Topic, and NER heads protected by Synaptic Intelligence.",
            "category": "Continual Learning",
            "badge": "Arena Ready",
            "config": {
                "embed_dim": 256,
                "num_layers": 4,
                "num_heads": 4,
                "head_dim": 64,
                "max_seq_len": 256,
                "use_flash_attention": True,
                "use_rotary_embedding": True,
                "use_amp": True,
                "use_si": True,
                "si_c": 1.0,
                "tasks": [
                    {
                        "task_name": "task_sentiment",
                        "task_type": "sentence_classification",
                        "num_classes": 2,
                        "max_len": 64,
                    },
                    {
                        "task_name": "task_topic",
                        "task_type": "sentence_classification",
                        "num_classes": 4,
                        "max_len": 64,
                    },
                    {
                        "task_name": "task_ner",
                        "task_type": "token_classification",
                        "num_classes": 5,
                        "max_len": 64,
                    },
                ],
            },
        },
        {
            "id": "adalora_lightweight",
            "name": "AdaLoRA Adaptive PEFT Model",
            "description": "Memory-efficient transformer utilizing adaptive SVD low-rank updates and sensitivity budget shrinking.",
            "category": "Parameter Efficient",
            "badge": "Ultra Low VRAM",
            "config": {
                "embed_dim": 256,
                "num_layers": 4,
                "num_heads": 4,
                "head_dim": 64,
                "max_seq_len": 256,
                "use_flash_attention": True,
                "use_rotary_embedding": True,
                "use_adalora": True,
                "adalora_init_rank": 16,
                "adalora_target_rank": 4,
                "tasks": [
                    {
                        "task_name": "task_topic",
                        "task_type": "sentence_classification",
                        "num_classes": 4,
                        "max_len": 64,
                    }
                ],
            },
        },
        {
            "id": "neural_storyteller",
            "name": "Neural Storyteller (Causal LM)",
            "description": "Autoregressive generative language model with causal attention masking and sliding window processing.",
            "category": "Generative",
            "badge": "Text Generation",
            "config": {
                "embed_dim": 384,
                "num_layers": 6,
                "num_heads": 6,
                "head_dim": 64,
                "max_seq_len": 512,
                "use_flash_attention": True,
                "use_rotary_embedding": True,
                "use_amp": True,
                "tasks": [
                    {
                        "task_name": "task_story_generation",
                        "task_type": "causal_lm",
                        "num_classes": 30522,
                        "max_len": 256,
                    }
                ],
            },
        },
    ]


@router.post("/estimate")
async def estimate_model_parameters(req: ModelBuildRequest):
    """Calculate parameter breakdown and memory footprint."""
    embed_params = req.vocab_size * req.embed_dim
    pos_params = 0 if req.use_rotary_embedding else (req.max_seq_len * req.embed_dim)

    # Per layer: MHA + FFN + LayerNorms
    mha_params = (4 * req.embed_dim * req.embed_dim) + (4 * req.embed_dim)
    ffn_dim = req.embed_dim * req.ff_mult
    ffn_params = (2 * req.embed_dim * ffn_dim) + req.embed_dim + ffn_dim
    norm_params = 4 * req.embed_dim

    layer_params = mha_params + ffn_params + norm_params
    encoder_params = req.num_layers * layer_params

    # Task heads
    head_breakdowns = []
    total_head_params = 0
    for task in (req.tasks or []):
        if isinstance(task, dict):
            t_name = str(task.get("task_name") or task.get("name") or "task")
            t_type = str(task.get("task_type") or task.get("strategy") or "sentence_classification").lower()
            num_c = int(task.get("num_classes") or len(task.get("classes") or []) or 2)
        else:
            t_name = getattr(task, "task_name", "task")
            t_type = str(getattr(task, "task_type", "sentence_classification") or getattr(task, "strategy", "sentence_classification")).lower()
            num_c = int(getattr(task, "num_classes", 2) or 2)

        if t_type in ("sentence_classification", "token_classification"):
            h_params = (req.embed_dim * num_c) + num_c
        elif t_type == "causal_lm":
            h_params = (req.embed_dim * (req.vocab_size or 30522))
        else:
            h_params = (req.embed_dim * 64)

        total_head_params += h_params
        head_breakdowns.append({
            "task_name": t_name,
            "task_type": t_type,
            "params": h_params,
        })

    total_params = embed_params + pos_params + encoder_params + total_head_params

    # Memory estimate (FP32 vs FP16)
    fp32_mb = round((total_params * 4) / (1024**2), 2)
    fp16_mb = round((total_params * 2) / (1024**2), 2)

    return {
        "total_parameters": total_params,
        "total_parameters_formatted": f"{total_params:,}",
        "embedding_params": embed_params,
        "encoder_params": encoder_params,
        "head_params_total": total_head_params,
        "head_breakdowns": head_breakdowns,
        "memory_footprint": {
            "fp32_mb": fp32_mb,
            "fp16_amp_mb": fp16_mb,
        },
        "python_code": generate_dytr_python_code(req),
    }


def generate_dytr_python_code(req: ModelBuildRequest) -> str:
    """Generate standalone executable Python script for DyTR based on the configuration."""
    tasks_code = ""
    for t in (req.tasks or []):
        if isinstance(t, dict):
            t_name = str(t.get("task_name") or t.get("name") or "task_default")
            raw_type = str(t.get("task_type") or t.get("strategy") or "sentence_classification").lower()
            num_classes = t.get("num_classes") or len(t.get("classes") or []) or 2
        else:
            t_name = str(getattr(t, "task_name", "task_default") or "task_default")
            raw_type = str(getattr(t, "task_type", None) or getattr(t, "strategy", "sentence_classification") or "sentence_classification").lower()
            num_classes = getattr(t, "num_classes", None) or 2

        strat_val = (
            "sentence_classification"
            if raw_type == "sentence_classification"
            else "token_classification"
            if raw_type == "token_classification"
            else "causal_lm"
        )
        tasks_code += f"""        "{t_name}": {{
            "strategy": "{strat_val}",
            "num_labels": {num_classes},
        }},
"""

    code = f"""# ==============================================================================
# DyTR Generated Training Script
# Generated automatically by DyTrStudio
# Author: Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
# ==============================================================================

import torch
from dytr.core.config import ModelConfig, TrainingStrategy
from dytr.core.model import DynamicTransformer

# 1. Define Model Architecture Configuration
config = ModelConfig(
    vocab_size={req.vocab_size},
    embed_dim={req.embed_dim},
    num_layers={req.num_layers},
    num_heads={req.num_heads},
    head_dim={req.head_dim or (req.embed_dim // req.num_heads)},
    ff_mult={req.ff_mult},
    max_seq_len={req.max_seq_len},
    dropout={req.dropout},
    use_flash_attention={req.use_flash_attention},
    use_rotary_embedding={req.use_rotary_embedding},
    gradient_checkpointing={req.gradient_checkpointing},
    tasks={{
{tasks_code}    }},
)

# 2. Instantiate Neural Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DynamicTransformer(config).to(device)
print(f"DyTR Model Initialized successfully on {{device}}!")
print(f"Total Parameters: {{sum(p.numel() for p in model.parameters()):,}}")
"""
    return code


@router.get("/download/checkpoint")
async def download_model_checkpoint(format: str = "pt"):
    """
    Download the real, genuine trained model checkpoint weights (.pt or .safetensors).
    Returns the real multi-megabyte binary tensor file.
    """
    from fastapi.responses import FileResponse
    from server.services.dytr_bridge import get_or_create_checkpoint_file

    file_path = get_or_create_checkpoint_file(format_type=format)
    filename = f"dytr_model_weights.{'safetensors' if format == 'safetensors' else 'pt'}"
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

