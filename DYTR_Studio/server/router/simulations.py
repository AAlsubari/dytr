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
Educational Simulation Router for DyTrStudio Glassroom.
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import Optional

from server.services.dytr_bridge import (
    get_system_capabilities,
    compute_rope_simulation,
    compute_continual_learning_sandbox,
    compute_multitask_lora_simulation,
    compute_adalora_svd_simulation,
    tokenize_with_subword_alignment,
)

router = APIRouter(prefix="/simulations", tags=["simulations"])


@router.get("/capabilities")
async def get_capabilities():
    """Return host hardware and DyTR computational capabilities."""
    return get_system_capabilities()


@router.get("/rope")
async def simulate_rope(
    token_index: int = Query(1, ge=0, le=512, description="Token position m in sequence"),
    dim: int = Query(16, ge=4, le=128, description="Subspace dimension (must be even)"),
    base: float = Query(10000.0, gt=0, description="Rotary base frequency"),
):
    """Simulate Rotary Position Embedding (RoPE) 2D vector rotation."""
    return compute_rope_simulation(token_index=token_index, dim=dim, base=base)


@router.get("/continual-learning")
async def simulate_continual_learning(
    use_ewc: bool = Query(True, description="Enable Elastic Weight Consolidation (EWC) protection"),
    ewc_lambda: float = Query(1000.0, ge=0.0, le=10000.0, description="EWC regularization penalty lambda"),
    use_replay: bool = Query(False, description="Enable Experience Replay buffer"),
    replay_buffer_size: int = Query(1000, ge=100, le=10000, description="Replay buffer capacity"),
    num_steps: int = Query(30, ge=10, le=100, description="Simulation steps"),
    use_si: Optional[bool] = Query(None, description="Backward compatibility for SI"),
    si_c: Optional[float] = Query(None, description="Backward compatibility for SI c"),
):
    """Simulate sequential 2-task learning with dytr EWC and Replay memory shields."""
    return compute_continual_learning_sandbox(
        use_ewc=use_ewc,
        ewc_lambda=ewc_lambda,
        use_replay=use_replay,
        replay_buffer_size=replay_buffer_size,
        num_steps=num_steps,
        use_si=use_si,
        si_c=si_c,
    )


@router.get("/lora-multitask")
@router.get("/adalora-svd")
async def simulate_lora(
    rank: int = Query(8, ge=1, le=64, description="LoRA adapter rank r"),
    alpha: float = Query(16.0, ge=1.0, le=128.0, description="LoRA scaling factor alpha"),
    d_in: int = Query(256, ge=32, le=1024, description="Input dimension"),
    d_out: int = Query(256, ge=32, le=1024, description="Output dimension"),
    num_tasks: int = Query(2, ge=1, le=8, description="Number of distinct multi-task adapters"),
    init_rank: Optional[int] = Query(None, description="Backward compat"),
    target_rank: Optional[int] = Query(None, description="Backward compat"),
):
    """Simulate dytr MultiTaskLoRAWrapper matrix decomposition and parameter efficiency."""
    return compute_multitask_lora_simulation(
        rank=rank,
        alpha=alpha,
        d_in=d_in,
        d_out=d_out,
        num_tasks=num_tasks,
        init_rank=init_rank,
        target_rank=target_rank,
    )


class TokenizeRequest(BaseModel):
    text: str = Field(default="Barack Obama visited the United Nations in New York")
    task_type: str = Field(default="token_classification")


@router.post("/tokenize-align")
async def simulate_tokenize_alignment(req: TokenizeRequest):
    """Simulate subword tokenization, alignment, and -100 loss masking."""
    return tokenize_with_subword_alignment(text=req.text, task_type=req.task_type)
