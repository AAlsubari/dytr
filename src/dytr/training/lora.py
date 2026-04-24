import math
import os
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dytr.core.model import DynamicTransformer
from dytr.core.config import ModelConfig, TaskConfig, TrainingStrategy
from dytr.training.data import MultiTaskDataset, TaskAwareBatchSampler, collate_fn


class LoRALayer(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.enabled = True

    def forward(self, x: torch.Tensor):
        result = self.linear(x)
        if not self.enabled:
            return result
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return result + self.dropout(lora_out)

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def merge(self):
        merged_weight = self.linear.weight.data + (self.lora_B @ self.lora_A) * self.scaling
        new_linear = nn.Linear(self.linear.in_features, self.linear.out_features, bias=self.linear.bias is not None)
        new_linear.weight.data = merged_weight
        if self.linear.bias is not None:
            new_linear.bias.data = self.linear.bias.data
        return new_linear


class TaskSpecificLoRALayer(nn.Module):
    def __init__(self, linear_layer: nn.Linear, task_name: str, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.linear = linear_layer
        self.task_name = task_name
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.enabled = True

    def forward(self, x: torch.Tensor):
        result = self.linear(x)
        if not self.enabled:
            return result
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return result + self.dropout(lora_out)


class LoRAWrapper:
    def __init__(self, model: DynamicTransformer, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1, target_modules: List[str] = None):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
        self.lora_layers = []
        self._apply_lora_to_model()
        self._freeze_base_model()

    def _apply_lora_to_model(self):
        for module_name, module in self.model.named_modules():
            if any(target in module_name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    lora = LoRALayer(module, self.rank, self.alpha, self.dropout)
                    self.lora_layers.append((module_name, lora))
                    self._replace_module(module_name, module, lora)

    def _replace_module(self, module_name: str, original_module: nn.Linear, lora_module: LoRALayer):
        parent_name, child_name = self._get_parent_and_child(module_name)
        parent = self._get_module_by_path(parent_name)
        setattr(parent, child_name, lora_module)

    def _get_parent_and_child(self, module_name: str):
        parts = module_name.split('.')
        child_name = parts[-1]
        parent_name = '.'.join(parts[:-1]) if len(parts) > 1 else ''
        return parent_name, child_name

    def _get_module_by_path(self, path: str):
        if not path:
            return self.model
        module = self.model
        for part in path.split('.'):
            module = getattr(module, part)
        return module

    def _freeze_base_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for _, lora in self.lora_layers:
            lora.lora_A.requires_grad = True
            lora.lora_B.requires_grad = True

    def get_trainable_parameters(self):
        params = []
        for _, lora in self.lora_layers:
            params.append(lora.lora_A)
            params.append(lora.lora_B)
        return params

    def merge_all_lora_weights(self):
        for module_name, lora in self.lora_layers:
            parent_name, child_name = self._get_parent_and_child(module_name)
            parent = self._get_module_by_path(parent_name)
            merged_linear = lora.merge()
            setattr(parent, child_name, merged_linear)


class MultiTaskLoRAWrapper:
    def __init__(self, model: DynamicTransformer, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1,target_modules: List[str] = None):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
        self.task_lora_layers = {}
        self._analyze_model_architecture()
        self._apply_lora_to_trainable_components()

    def _analyze_model_architecture(self):
        self.trainable_status = {}
        for name, param in self.model.named_parameters():
            self.trainable_status[name] = param.requires_grad
        self.task_heads = list(self.model.task_heads.keys())
        self.decoders = list(self.model.decoders.keys())
        self.encoder_trainable = any('encoder' in n and p.requires_grad for n, p in self.model.named_parameters())
        self.shared_embedding_trainable = any('shared_embedding' in n and p.requires_grad for n, p in self.model.named_parameters())

    def _apply_lora_to_trainable_components(self):
        #target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(target in module_name for target in self.target_modules):
                continue
            should_apply = self._should_apply_lora(module_name)
            if not should_apply:
                continue
            self.task_lora_layers[module_name] = {}
            for task_name in self.model.current_tasks.keys():
                if not self._is_task_relevant(module_name, task_name):
                    continue
                lora = TaskSpecificLoRALayer(module, task_name, self.rank, self.alpha, self.dropout)
                self.task_lora_layers[module_name][task_name] = lora
            if self.task_lora_layers[module_name]:
                self._replace_module_with_lora(module_name, module)

    def _should_apply_lora(self, module_name: str) -> bool:
        if 'encoder' in module_name:
            return self.encoder_trainable
        if 'shared_embedding' in module_name:
            return self.shared_embedding_trainable
        if 'decoders' in module_name:
            parts = module_name.split('.')
            decoder_name = parts[1] if len(parts) > 1 else None
            if decoder_name and decoder_name in self.decoders:
                return True
        if 'task_heads' in module_name:
            parts = module_name.split('.')
            head_name = parts[1] if len(parts) > 1 else None
            if head_name and head_name in self.task_heads:
                return True
        return False

    def _is_task_relevant(self, module_name: str, task_name: str) -> bool:
        if 'decoders' in module_name:
            parts = module_name.split('.')
            decoder_name = parts[1] if len(parts) > 1 else None
            return decoder_name == task_name
        if 'task_heads' in module_name:
            parts = module_name.split('.')
            head_name = parts[1] if len(parts) > 1 else None
            return head_name == task_name
        return True

    def _replace_module_with_lora(self, module_name: str, original_module: nn.Linear):
        parent_name, child_name = self._get_parent_and_child(module_name)
        parent = self._get_module_by_path(parent_name)
        
        class MultiTaskLinear(nn.Module):
            def __init__(self, original, lora_dict):
                super().__init__()
                self.original = original
                self.lora_dict = lora_dict
                self.current_task = None
            
            def forward(self, x):
                result = self.original(x)
                if self.current_task and self.current_task in self.lora_dict:
                    lora = self.lora_dict[self.current_task]
                    lora_out = (x @ lora.lora_A.T) @ lora.lora_B.T * lora.scaling
                    result = result + lora_out
                return result
        
        multi_linear = MultiTaskLinear(original_module, self.task_lora_layers[module_name])
        setattr(parent, child_name, multi_linear)

    def _get_parent_and_child(self, module_name: str):
        parts = module_name.split('.')
        child_name = parts[-1]
        parent_name = '.'.join(parts[:-1]) if len(parts) > 1 else ''
        return parent_name, child_name

    def _get_module_by_path(self, path: str):
        if not path:
            return self.model
        module = self.model
        for part in path.split('.'):
            module = getattr(module, part)
        return module

    def set_current_task(self, task_name: str):
        for module_name, task_loras in self.task_lora_layers.items():
            parent_name, child_name = self._get_parent_and_child(module_name)
            parent = self._get_module_by_path(parent_name)
            if hasattr(parent, child_name):
                multi_linear = getattr(parent, child_name)
                multi_linear.current_task = task_name

    def get_trainable_parameters(self):
        params = []
        for module_name, task_loras in self.task_lora_layers.items():
            parent_name, child_name = self._get_parent_and_child(module_name)
            parent = self._get_module_by_path(parent_name)
            if not hasattr(parent, child_name):
                continue
            multi_linear = getattr(parent, child_name)
            for task_name, lora in task_loras.items():
                if task_name in multi_linear.lora_dict:
                    params.append(multi_linear.lora_dict[task_name].lora_A)
                    params.append(multi_linear.lora_dict[task_name].lora_B)
        return params

    def merge_task_weights(self, task_name: str):
        for module_name, task_loras in self.task_lora_layers.items():
            parent_name, child_name = self._get_parent_and_child(module_name)
            parent = self._get_module_by_path(parent_name)
            if not hasattr(parent, child_name):
                continue
            multi_linear = getattr(parent, child_name)
            if task_name in multi_linear.lora_dict:
                lora = multi_linear.lora_dict[task_name]
                merged_weight = multi_linear.original.weight.data + (lora.lora_B @ lora.lora_A) * lora.scaling
                multi_linear.original.weight.data = merged_weight


class ProgressiveLoRATrainer:
    def __init__(self, model: DynamicTransformer, config: ModelConfig, exp_dir: str, initial_rank: int = 4, final_rank: int = 16):
        self.model = model
        self.config = config
        self.exp_dir = exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        self.initial_rank = initial_rank
        self.final_rank = final_rank
        self.current_rank = initial_rank
        self.lora_wrapper = None
        self.val_loader=None
        self.best_val_loss = float('inf')

    def train(self, task_configs: List[TaskConfig], train_datasets: Dict, val_datasets: Dict):
        for task_config in task_configs:
            if task_config.task_name not in self.model.current_tasks:
                self.model.add_task(task_config)
        
        self._phase_1_warmup(task_configs, train_datasets, val_datasets)
        self._phase_2_expansion(task_configs, train_datasets, val_datasets)
        self._phase_3_hybrid(task_configs, train_datasets, val_datasets)
        self._phase_4_native(task_configs, train_datasets, val_datasets)
        
        return self.model

    def _phase_1_warmup(self, task_configs: List[TaskConfig], train_datasets: Dict, val_datasets: Dict):
        print("\n" + "="*60)
        print("Phase 1: Warmup with Low-Rank LoRA")
        print("="*60)
        
        self.current_rank = self.initial_rank
        self.lora_wrapper = LoRAWrapper(self.model, rank=self.current_rank, alpha=self.current_rank * 2)
        
        train_dataset = MultiTaskDataset(train_datasets, self.model.tokenizer, task_configs)
        train_sampler = TaskAwareBatchSampler(train_dataset, self.config.per_device_train_batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0)
        
        trainable_params = self.lora_wrapper.get_trainable_parameters()
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        original_epochs = self.config.num_train_epochs
        self.config.num_train_epochs = 2
        
        self._run_training_phase(task_configs, train_datasets, val_datasets, train_loader, optimizer, "Warmup")
        
        self.config.num_train_epochs = original_epochs

    def _phase_2_expansion(self, task_configs: List[TaskConfig], train_datasets: Dict, val_datasets: Dict):
        print("\n" + "="*60)
        print("Phase 2: Expansion with Increased LoRA Rank")
        print("="*60)
        
        self.lora_wrapper.merge_all_lora_weights()
        
        self.current_rank = self.final_rank
        self.lora_wrapper = LoRAWrapper(self.model, rank=self.current_rank, alpha=self.current_rank * 2)
        
        train_dataset = MultiTaskDataset(train_datasets, self.model.tokenizer, task_configs)
        train_sampler = TaskAwareBatchSampler(train_dataset, self.config.per_device_train_batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0)
        
        trainable_params = self.lora_wrapper.get_trainable_parameters()
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate * 0.5, weight_decay=self.config.weight_decay)
        
        original_epochs = self.config.num_train_epochs
        self.config.num_train_epochs = 3
        
        self._run_training_phase(task_configs, train_datasets, val_datasets, train_loader, optimizer, "Expansion")
        
        self.config.num_train_epochs = original_epochs

    def _phase_3_hybrid(self, task_configs: List[TaskConfig], train_datasets: Dict, val_datasets: Dict):
        print("\n" + "="*60)
        print("Phase 3: Hybrid (LoRA on Encoder Only)")
        print("="*60)
        
        self.lora_wrapper.merge_all_lora_weights()
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = True
        
        train_dataset = MultiTaskDataset(train_datasets, self.model.tokenizer, task_configs)
        train_sampler = TaskAwareBatchSampler(train_dataset, self.config.per_device_train_batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0)
        
        encoder_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'encoder' in name:
                    encoder_params.append(param)
                else:
                    other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.config.learning_rate * 0.1},
            {'params': other_params, 'lr': self.config.learning_rate}
        ], weight_decay=self.config.weight_decay)
        
        original_epochs = self.config.num_train_epochs
        self.config.num_train_epochs = 5
        
        self._run_training_phase(task_configs, train_datasets, val_datasets, train_loader, optimizer, "Hybrid")
        
        self.config.num_train_epochs = original_epochs

    def _phase_4_native(self, task_configs: List[TaskConfig], train_datasets: Dict, val_datasets: Dict):
        print("\n" + "="*60)
        print("Phase 4: Full Native Training")
        print("="*60)
        
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        
        train_dataset = MultiTaskDataset(train_datasets, self.model.tokenizer, task_configs)
        train_sampler = TaskAwareBatchSampler(train_dataset, self.config.per_device_train_batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate * 0.1, weight_decay=self.config.weight_decay)
        
        self._run_training_phase(task_configs, train_datasets, val_datasets, train_loader, optimizer, "Native")

    def _run_training_phase(self, task_configs: List[TaskConfig], train_datasets: Dict, val_datasets: Dict, train_loader, optimizer, phase_name):
        if val_datasets:
            val_dataset = MultiTaskDataset(val_datasets, self.model.tokenizer, task_configs)
            val_sampler = TaskAwareBatchSampler(val_dataset, self.config.per_device_eval_batch_size, shuffle=False)
            self.val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn, num_workers=0)
            
        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"{phase_name} Epoch {epoch+1}/{self.config.num_train_epochs}")
            
            for batch in pbar:
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    task_name=batch["task_name"][0],
                    labels=batch.get("labels")
                )
                
                if outputs is not None and "loss" in outputs and outputs["loss"] is not None:
                    loss = outputs["loss"]
                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.item()
                        num_batches += 1
                        pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
            
            if num_batches > 0:
                print(f"\n{phase_name} Epoch {epoch+1} average loss: {total_loss/num_batches:.4f}")
                if self.val_loader is not None:
                    val_loss = self._validate()
                    print(f"validation loss:{val_loss}")
                
                    if val_loss is not None and val_loss < self.best_val_loss:
                        print("best val loss --->",val_loss)
                        self.best_val_loss = val_loss
        
        final_path = os.path.join(self.exp_dir, f"{phase_name.lower()}_model.pt")
        self.model.save_model(final_path)
    def _validate(self):
        self.model.eval()
        total_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                task_name = batch["task_name"][0]
                
                
                
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    task_name=task_name,
                    labels=batch.get("labels")
                )
                
                if outputs is not None and "loss" in outputs and outputs["loss"] is not None:
                    if torch.isfinite(outputs["loss"]):
                        total_loss += outputs["loss"].item()
                        val_batches += 1
        
        if val_batches > 0:
            return total_loss / val_batches
        return None


class LoRATrainer:
    def __init__(self, model: DynamicTransformer, config: ModelConfig, exp_dir: str, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1, mode: str = "single"):
        self.model = model
        self.config = model.config
        self.exp_dir = exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.mode = mode
        self.lora_wrapper = None
        self.best_val_loss = float('inf')

    def train(self, task_configs: List[TaskConfig], train_datasets: Dict, val_datasets: Dict):
        for task_config in task_configs:
            if task_config.task_name not in self.model.current_tasks:
                self.model.add_task(task_config)
        
        if self.mode == "single":
            self.lora_wrapper = LoRAWrapper(self.model, self.rank, self.alpha, self.dropout)
        elif self.mode == "multi":
            self.lora_wrapper = MultiTaskLoRAWrapper(self.model, self.rank, self.alpha, self.dropout)
        elif self.mode == "progressive":
            progressive_trainer = ProgressiveLoRATrainer(self.model, self.config, self.exp_dir, self.rank, self.rank * 2)
            return progressive_trainer.train(task_configs, train_datasets, val_datasets)
        else:
            self.lora_wrapper = LoRAWrapper(self.model, self.rank, self.alpha, self.dropout)
        
        train_dataset = MultiTaskDataset(train_datasets, self.model.tokenizer, task_configs)
        val_dataset = None
        val_loader = None
        print(len(val_datasets))
        if val_datasets:
            val_dataset = MultiTaskDataset(val_datasets, self.model.tokenizer, task_configs)
            val_sampler = TaskAwareBatchSampler(val_dataset, self.config.per_device_eval_batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn, num_workers=0)
            
        
        train_sampler = TaskAwareBatchSampler(train_dataset, self.config.per_device_train_batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0)
        
        
        
        trainable_params = self.lora_wrapper.get_trainable_parameters()
        
        print(f"\nLoRA Training Mode: {self.mode}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"Total model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Reduction: {(1 - sum(p.numel() for p in trainable_params) / sum(p.numel() for p in self.model.parameters())) * 100:.2f}%")
        
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_train_epochs}")
            
            for batch in pbar:
                task_name = batch["task_name"][0]
                
                if self.mode == "multi" and hasattr(self.lora_wrapper, 'set_current_task'):
                    self.lora_wrapper.set_current_task(task_name)
                
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    task_name=task_name,
                    labels=batch.get("labels")
                )
                
                if outputs is not None and "loss" in outputs and outputs["loss"] is not None:
                    loss = outputs["loss"]
                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.item()
                        num_batches += 1
                        pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
            
            if num_batches > 0:
                print(f"\nEpoch {epoch+1} average loss: {total_loss/num_batches:.4f}")
            #print("@"*100)
            #print(len(val_loader))
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                print(f"validation loss:{val_loss}")
                
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_model()
        
        if self.mode == "single":
            self.lora_wrapper.merge_all_lora_weights()
        
        final_path = os.path.join(self.exp_dir, "final_model.pt")
        self.model.save_model(final_path)
        
        return self.model

    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                task_name = batch["task_name"][0]
                
                if self.mode == "multi" and hasattr(self.lora_wrapper, 'set_current_task'):
                    self.lora_wrapper.set_current_task(task_name)
                
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    task_name=task_name,
                    labels=batch.get("labels")
                )
                
                if outputs is not None and "loss" in outputs and outputs["loss"] is not None:
                    if torch.isfinite(outputs["loss"]):
                        total_loss += outputs["loss"].item()
                        val_batches += 1
        
        if val_batches > 0:
            return total_loss / val_batches
        return None

    def _save_model(self):
        save_path = os.path.join(self.exp_dir, "best_model.pt")
        os.makedirs(self.exp_dir, exist_ok=True)
        self.model.save_model(save_path)
        print(f"Best model saved to {save_path}")
