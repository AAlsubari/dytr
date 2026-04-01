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
Elastic Weight Consolidation (EWC) for continual learning.

This module implements EWC to prevent catastrophic forgetting when training
on sequential tasks.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""

from typing import Dict

import torch
import torch.nn as nn


class EWC:
    """
    Elastic Weight Consolidation for continual learning.

    EWC computes the Fisher information matrix for important parameters
    and adds a penalty to prevent them from changing too much when learning
    new tasks.

    Args:
        model: The model to apply EWC to
        task_name: Name of the task this EWC instance is for
        lambda_param: Regularization strength
    """

    def __init__(self, model: nn.Module, task_name: str, lambda_param: float = 1000.0):
        self.model = model
        self.task_name = task_name
        self.lambda_param = lambda_param
        self.params = {
            n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad
        }
        self.fisher: Dict[str, torch.Tensor] = {}

    def compute_fisher(self, dataloader, device):
        """
        Compute Fisher information matrix using the given dataloader.

        Args:
            dataloader: DataLoader containing samples from the task
            device: Device to run computation on
        """
        # Initialize Fisher matrix
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.fisher[n] = torch.zeros_like(p)

        self.model.eval()

        for batch in dataloader:
            # Only process batches for this task
            if batch["task_name"][0] != self.task_name:
                continue

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            self.model.zero_grad()

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                task_name=self.task_name,
                labels=batch.get("labels"),
            )

            loss = outputs["loss"]
            if loss is not None:
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        self.fisher[n] += p.grad.detach() ** 2

        # Normalize Fisher matrix
        for n in self.fisher:
            self.fisher[n] /= len(dataloader)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC penalty for the current model.

        Args:
            model: Current model state

        Returns:
            EWC penalty loss
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.params:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.lambda_param * loss
