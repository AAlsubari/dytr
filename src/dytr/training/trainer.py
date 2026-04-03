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
Trainer class for multi-task continual learning.

This module provides the Trainer class which handles training,
validation, and continual learning features like EWC and experience replay.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""



import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
#import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dytr.core.config import ModelConfig, TaskConfig
from dytr.core.model import DynamicTransformer
from dytr.training.data import MultiTaskDataset, TaskAwareBatchSampler, collate_fn
from dytr import EWC
from dytr.utils.logging import get_logger

class Trainer:
    """
    Trainer class for multi-task continual learning.

    Handles training across multiple tasks with support for:
    - EWC (Elastic Weight Consolidation)
    - Experience replay
    - Learning rate scheduling
    - Gradient accumulation
    - Early stopping
    - Model checkpointing

    Args:
        model: DynamicTransformer model
        config: Model configuration
        exp_dir: Directory for saving experiment outputs
    """

    def __init__(self, model: DynamicTransformer, config: ModelConfig, exp_dir: str):
        self.logger = get_logger(__name__)
        self.model = model
        self.config = config
        self.device = config.device
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = model.best_val_loss
        self.best_metrics = {}
        self.patience_counter = 0
        self.loss_history = []
        self.num_labels_per_task={}
        self.val_loss_history = []
        self.metrics_history = defaultdict(list)
        self.logger.info("Trainer initialized")
        #self.logger.debug(f"Experiment directory: {exp_dir}")
        self.logger.debug("Device: %s ",self.config.device)

    def count_parameters(self):
        """Count and display model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info("  Model size: %s parameters",total)
        self.logger.info("  Trainable: %s" ,trainable)
        self.logger.info("\nParameter breakdown:")

        for name, module in self.model.task_heads.items():
            self.logger.info("    Head %s: %s",name, sum(p.numel() for p in module.parameters()))

        for name, module in self.model.decoders.items():
            self.logger.info("    Decoder %s: %s",name, sum(p.numel() for p in module.parameters()))

        self.logger.info(f"    Shared encoder: {sum(p.numel() for p in self.model.encoder.parameters()):,}")
        self.logger.info(
            f"    Shared embedding: {sum(p.numel() for p in self.model.shared_embedding.parameters()):,}"
        )

        return total
    def freeze_all(self):
        """
        Freeze all parameters in the model.
        Used as initial state before selectively unfreezing tasks.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        self.logger.info("  Frozen all model parameters")
    
    def unfreeze_shared_encoder(self):
        """
        Unfreeze the shared encoder and embedding layers.
        These are shared across all tasks and should be trainable for all tasks.
        """
        # Unfreeze shared embedding
        for param in self.model.shared_embedding.parameters():
            param.requires_grad = True
        
        # Unfreeze encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        
        self.logger.info("  Unfrozen shared encoder and embeddings")
    
    def unfreeze_task_components(self, task_name: str):
        """
        Unfreeze task-specific components for a given task.
        
        Args:
            task_name: Name of the task to unfreeze
        """
        # Unfreeze task head if it exists
        if task_name in self.model.task_heads:
            for param in self.model.task_heads[task_name].parameters():
                param.requires_grad = True
            self.logger.info(f"  Unfrozen head for task: {task_name}")
        
        # Unfreeze decoder if it exists
        if task_name in self.model.decoders:
            for param in self.model.decoders[task_name].parameters():
                param.requires_grad = True
            self.logger.info(f"  Unfrozen decoder for task: {task_name}")
    
    def unfreeze_for_tasks(self, task_names: list, unfreeze_shared: bool = True):
        """
        Unfreeze model components for specific tasks.
        
        Args:
            task_names: List of tasks to unfreeze
            unfreeze_shared: Whether to unfreeze shared encoder (default: True)
        """
        # Unfreeze shared components first (if needed)
        if unfreeze_shared:
            self.unfreeze_shared_encoder()
        
        # Unfreeze each task's specific components
        for task_name in task_names:
            self.unfreeze_task_components(task_name)
    
    
    def train(self, task_configs: List[TaskConfig], train_datasets: Dict, val_datasets: Dict):
        """
        Train the model on multiple tasks.

        Args:
            task_configs: List of task configurations
            train_datasets: Dictionary of training datasets per task
            val_datasets: Dictionary of validation datasets per task

        Returns:
            Trained model
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting training session")
        self.logger.info("=" * 60)
        training_task_names = list(train_datasets.keys())
        self.logger.debug(f"Task configs: {[tc.task_name for tc in task_configs]}")
        self.logger.debug(f"Train datasets: {training_task_names}")
        self.logger.debug(f"Val datasets: {list(val_datasets.keys())}")
        # Add tasks to model
        self.freeze_all()
        for task_config in task_configs:
            self.model.add_task(task_config)
            if task_config.task_name not in self.num_labels_per_task:
                self.num_labels_per_task[task_config.task_name]=task_config.num_labels
        
        self.unfreeze_for_tasks(training_task_names)
        if len(train_datasets.keys())<1:
            self.logger.error("No training datasets found for tasks!")
            return self.model
        
        self.logger.info("Model parameters after adding tasks:")
        self.count_parameters()

        # Create multi-task datasets
        train_dataset = MultiTaskDataset(train_datasets, self.model.tokenizer, task_configs)
        val_dataset = MultiTaskDataset(val_datasets, self.model.tokenizer, task_configs)

        # Create samplers and dataloaders
        train_sampler = TaskAwareBatchSampler(
            train_dataset, self.config.per_device_train_batch_size, shuffle=True
        )
        use_fixed_padding = self.config.use_replay and len(self.model.replay_buffer.buffer) > 0

        def train_collate(batch):
            if use_fixed_padding:
                # Use fixed max length from config
                return collate_fn(batch, fixed_max_len=self.config.max_seq_len)
            else:
                # Use dynamic padding
                return collate_fn(batch)

        train_loader = DataLoader(
            train_dataset, batch_sampler=train_sampler, collate_fn=train_collate, num_workers=0
        )
        
        if len(val_dataset) > 0:
            val_sampler = TaskAwareBatchSampler(
                val_dataset, self.config.per_device_eval_batch_size, shuffle=False
            )
            val_loader = DataLoader(
                val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn, num_workers=0
            )
        else:
            val_loader = None
            self.logger.warning("  Warning: No validation data provided")

        # Setup optimizer with task-specific learning rates
        param_groups = []
        assigned = set()

        # Task heads parameters
        head_params = []
        for th in self.model.task_heads.values():
            for p in th.parameters():
                if id(p) not in assigned:
                    head_params.append(p)
                    assigned.add(id(p))
        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": self.config.learning_rate * self.config.head_lr_mult,
                    "name": "task_heads",
                }
            )

        # Decoder parameters
        decoder_params = []
        for d in self.model.decoders.values():
            for p in d.parameters():
                if id(p) not in assigned:
                    decoder_params.append(p)
                    assigned.add(id(p))
        if decoder_params:
            param_groups.append(
                {
                    "params": decoder_params,
                    "lr": self.config.learning_rate * self.config.decoder_lr_mult,
                    "name": "decoders",
                }
            )

        # Shared parameters
        shared_params = []
        for p in self.model.parameters():
            if id(p) not in assigned:
                shared_params.append(p)
                assigned.add(id(p))
        if shared_params:
            param_groups.append(
                {
                    "params": shared_params,
                    "lr": self.config.learning_rate * self.config.shared_lr_mult,
                    "name": "shared",
                }
            )

        # Optimizer and scheduler
        optimizer = optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
        total_steps = len(train_loader) * self.config.num_train_epochs

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, total_steps - self.config.warmup_steps
            )
            return max(
                self.config.min_learning_rate / self.config.learning_rate,
                0.5 * (1 + math.cos(math.pi * progress)),
            )

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        best_model_path = os.path.join(self.exp_dir, "best_model.pt")
        steps=0




        # Training loop
        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            task_losses = defaultdict(list)
            epoch_metrics = defaultdict(lambda: defaultdict(list))

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_train_epochs}")

            for batch_idx, batch in enumerate(pbar):
                try:
                    task_name = batch["task_name"][0]
                    strategy=batch["strategy"][0]

                    # Move batch to device
                    
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    # Add replay samples if available
                    if self.config.use_replay and len(self.model.replay_buffer.buffer) > 0:
                        replay_samples = self.model.replay_buffer.sample(
                            self.config.per_device_train_batch_size // 4
                        )
                        if replay_samples:
                            
                            for replay in replay_samples:
                                if task_name in self.num_labels_per_task and replay["task_name"] in self.num_labels_per_task and self.num_labels_per_task[replay["task_name"]]!=self.num_labels_per_task[task_name]:
                                    break
                                elif task_name not in self.num_labels_per_task or replay["task_name"] not in self.num_labels_per_task:
                                    break
                                elif replay["strategy"]!=strategy:
                                    break
                                #print(replay["strategy"])
                                for k, v in replay.items():
                                    if isinstance(v, torch.Tensor):
                                        if batch[k].dim() == v.dim():
                                            # Same dimensions, can concatenate
                                            batch[k] = torch.cat(
                                                [batch[k], v.to(self.device)], dim=0
                                            )
                                        else:
                                            # Different dimensions, need to handle specially
                                            # print(f"  Warning: Cannot concatenate {k} with dim {batch[k].dim()} and {v.dim()}")
                                            # Skip this replay sample for this key
                                            pass
                                        # batch[k] = torch.cat([batch[k], v.to(self.device)], dim=0)

                    # Forward pass
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        task_name=task_name,
                        labels=batch.get("labels"),
                    )

                    if outputs is not None and "loss" in outputs and outputs["loss"] is not None:
                        loss = outputs["loss"]

                        # Add EWC penalty if enabled
                        if self.config.use_ewc and len(self.model.ewc_penalties) > 0:
                            ewc_loss = 0
                            for ts in self.model.ewc_penalties.keys():
                                if ts in training_task_names:
                                    continue
                                ewc_loss += self.model.ewc_penalties[ts].penalty(self.model)
                                
                            #for ewc in self.model.ewc_penalties.values():
                            #    ewc_loss += ewc.penalty(self.model)
                            loss = loss + self.config.ewc_lambda * ewc_loss

                        if torch.isfinite(loss):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config.max_grad_norm
                            )
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()

                            total_loss += loss.item()
                            num_batches += 1
                            steps+=1
                            self.loss_history.append(loss.item())
                            task_losses[task_name].append(loss.item())

                            # Collect metrics
                            if "metrics" in outputs and outputs["metrics"]:
                                for metric_name, value in outputs["metrics"].items():
                                    if isinstance(value, (int, float)):
                                        epoch_metrics[task_name][metric_name].append(value)
                                    elif isinstance(value, torch.Tensor):
                                        epoch_metrics[task_name][metric_name].append(value.item())
                        else:
                            self.logger.warning(f"  Warning: Non-finite loss at batch {batch_idx}: {loss}")

                    # Logging
                    if steps % self.config.logging_steps == 0 and num_batches > 0:
                        task_avg = {
                            task: sum(losses[-100:]) / len(losses[-100:])
                            for task, losses in task_losses.items()
                            if losses
                        }
                        pbar.set_postfix(
                            {"loss": f"{total_loss/num_batches:.4f}", "tasks": str(task_avg)}
                        )

                    # Validation during training (if steps strategy)
                    if (
                        self.config.evaluation_strategy == "steps"
                        and steps % self.config.validation_check_interval == 0
                        and num_batches > 0
                        and val_loader is not None
                    ):
                        val_results = self.validate(val_loader)
                        if val_results:
                            self._handle_validation(val_results, best_model_path)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        self.logger.warning(f"  Error in training batch: {e}")
                        continue
                    else:
                        raise e
                except Exception as e:
                    self.logger.warning(f"  Error in training batch: {e}")
                    continue

            # Epoch end
            if num_batches > 0:
                avg_epoch_loss = total_loss / num_batches
                print(f"\n  Epoch {epoch+1} average training loss: {avg_epoch_loss:.4f}")

                for task, metrics in epoch_metrics.items():
                    print(f"    {task} training metrics:")
                    for metric_name, values in metrics.items():
                        if values:
                            avg_value = sum(values) / len(values)
                            print(f"      {metric_name}: {avg_value:.4f}")

            # Validation at epoch end
            if val_loader is not None and self.config.evaluation_strategy != "steps":
                val_results = self.validate(val_loader)
                if val_results:
                    self._handle_validation(val_results, best_model_path)

                    if self.patience_counter >= self.config.patience:
                        print(f"  Early stopping triggered after {epoch+1} Validation")
                        response=input("Insert letter Y to stop the training: [Y/n]")
                        if response=='Y':
                            break
                        self.patience_counter=0
                        
                            
        
        self.logger.debug(f"\n  Training completed. Best validation loss: {self.best_val_loss:.4f}")
        if self.config.use_ewc:# and use_fixed_padding:
            for task_name in train_datasets.keys():
                if task_name in self.model.ewc_penalties: continue
                ewc = EWC(self.model, task_name, lambda_param=self.config.ewc_lambda)
                fisher_loader = DataLoader(train_datasets[task_name][0], batch_size=8, shuffle=True, collate_fn=collate_fn)
                ewc.compute_fisher(fisher_loader, self.config.device)
                self.model.ewc_penalties[task_name] = ewc
        return self.model

    def _handle_validation(self, val_results, best_model_path):
        """Handle validation results and model checkpointing."""
        val_loss = val_results["avg_loss"]
        val_metrics = val_results["metrics"]

        self.val_loss_history.append(val_loss)
        print(f"\n  Validation loss: {val_loss:.4f}")

        for task, metrics in val_metrics.items():
            print(f"    {task} validation:")
            for metric_name, value in metrics.items():
                if isinstance(value, dict):
                    print(f"      {metric_name}:")
                    for k, v in value.items():
                        print(f"        {k}: {v:.4f}")
                else:
                    print(f"      {metric_name}: {value:.4f}")

        if val_loss < self.best_val_loss:
            improvement = self.best_val_loss - val_loss
            self.model.steps_without_improvement = 0
            self.best_val_loss = val_loss
            self.model.best_val_loss = val_loss
            self.best_metrics = val_metrics
            self.patience_counter = 0
            self.model.save_model(best_model_path)
            self.logger.warning("  ✓ Best model saved improvement: %s",improvement)
        else:
            self.model.steps_without_improvement += 1
            self.patience_counter += 1
            self.logger.warning("  No improvement for %s epochs",self.patience_counter)

    def validate(self, val_loader):
        """
        Validate the model on validation data.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary with average loss and metrics per task
        """
        self.model.eval()
        total_loss = 0
        val_batches = 0
        task_metrics = defaultdict(lambda: defaultdict(list))

        with torch.no_grad():
            for batch in val_loader:
                task_name = batch["task_name"][0]

                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    task_name=task_name,
                    labels=batch.get("labels"),
                )

                if outputs is not None and "loss" in outputs and outputs["loss"] is not None:
                    if torch.isfinite(outputs["loss"]):
                        total_loss += outputs["loss"].item()
                        val_batches += 1

                        if "metrics" in outputs and outputs["metrics"]:
                            for metric_name, value in outputs["metrics"].items():
                                if isinstance(value, torch.Tensor):
                                    task_metrics[task_name][metric_name].append(value.cpu().item())
                                elif isinstance(value, dict):
                                    for sub_metric, sub_value in value.items():
                                        combined_key = f"{metric_name}_{sub_metric}"
                                        if isinstance(sub_value, (int, float, torch.Tensor)):
                                            if isinstance(sub_value, torch.Tensor):
                                                task_metrics[task_name][combined_key].append(
                                                    sub_value.item()
                                                )
                                            else:
                                                task_metrics[task_name][combined_key].append(
                                                    sub_value
                                                )
                                else:
                                    task_metrics[task_name][metric_name].append(value)

        if val_batches > 0:
            avg_loss = total_loss / val_batches
            aggregated_metrics = {}
            for task, metrics in task_metrics.items():
                aggregated_metrics[task] = {}
                for metric_name, values in metrics.items():
                    if values:
                        aggregated_metrics[task][metric_name] = sum(values) / len(values)
            return {"avg_loss": avg_loss, "metrics": aggregated_metrics}
        return None
