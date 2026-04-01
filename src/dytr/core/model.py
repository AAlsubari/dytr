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
Main DynamicTransformer model implementation.

This module contains the core DynamicTransformer class that manages multiple tasks,
task-specific heads and decoders, and handles different training strategies.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dytr.core.config import ModelConfig, TaskConfig, TrainingStrategy
from dytr.core.decoder import TransformerDecoder
from dytr.core.encoder import TransformerEncoder
from dytr.core.exporter import ModelExporter
from dytr.memory.ewc import EWC
from dytr.memory.replay import ReplayBuffer

from collections import defaultdict
import numpy as np




def calculate_f_scores(preds, labels, average='macro', beta=1.0):
    """
    Calculate F-beta score (F1 when beta=1, F0.5 when beta=0.5)
    
    Args:
        preds: Tensor of predictions
        labels: Tensor of ground truth labels
        average: 'macro', 'micro', 'weighted', or None
        beta: Beta parameter for F-beta score
    Returns:
        f_score: F-beta score
        per_class: Dictionary of per-class scores (if average is None)
    """
    
    
    if labels.numel() == 0:
        return 0.0, {}
    
    # Convert to CPU numpy for calculation
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Get unique classes
    classes = np.unique(np.concatenate([labels_np, preds_np]))
    n_classes = len(classes)
    
    if n_classes == 0:
        return 0.0, {}
    
    # Initialize confusion matrix components
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    
    for pred, label in zip(preds_np, labels_np):
        if pred == label:
            tp[label] += 1
        else:
            fp[pred] += 1
            fn[label] += 1
    
    # Calculate precision, recall, and f-beta for each class
    precision = {}
    recall = {}
    f_beta = {}
    
    for cls in classes:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        
        precision[cls] = p
        recall[cls] = r
        
        if p + r > 0:
            beta_sq = beta * beta
            f_beta[cls] = (1 + beta_sq) * (p * r) / ((beta_sq * p) + r)
        else:
            f_beta[cls] = 0.0
    
    if average == 'macro':
        return sum(f_beta.values()) / n_classes, f_beta
    elif average == 'weighted':
        # Count samples per class
        class_counts = defaultdict(int)
        for label in labels_np:
            class_counts[label] += 1
        total = len(labels_np)
        weighted_score = sum(f_beta[cls] * (class_counts[cls] / total) for cls in classes)
        return weighted_score, f_beta
    elif average == 'micro':
        # Micro averaging: aggregate all classes
        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        beta_sq = beta * beta
        micro_f = (1 + beta_sq) * (micro_p * micro_r) / ((beta_sq * micro_p) + micro_r) if (micro_p + micro_r) > 0 else 0.0
        return micro_f, f_beta
    else:
        return f_beta, f_beta



class DynamicTransformer(nn.Module):
    """
    Main Dynamic Transformer model supporting multi-task learning and continual learning.

    This model can handle multiple tasks with different training strategies simultaneously,
    with support for EWC (Elastic Weight Consolidation) and experience replay to prevent
    catastrophic forgetting.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.current_tasks = {}
        self.config = config

        # Initialize tokenizer (try HuggingFace first, fallback to SimpleTokenizer)
        try:
            from transformers import AutoTokenizer
            transformers_available = True
        except:
            transformers_available = False
            
        if transformers_available and not config.use_simple_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        else:
            
            from dytr.tokenization.download_manager import download_tokenizer_files
            from dytr.tokenization.simple_tokenizer import SimpleTokenizer

            if "vocab.json" in config.tokenizer_name :
                import json

                with open(config.tokenizer_name, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                self.tokenizer = SimpleTokenizer(vocab,add_tab_newline_vocab = config.add_tab_newline_vocab, tokenizer_type=config.tokenizer_type)
            else:
                try:
                    vocab, special_token_map = download_tokenizer_files(str(config.tokenizer_name))
                    self.tokenizer = SimpleTokenizer(vocab, special_token_map,add_tab_newline_vocab = config.add_tab_newline_vocab,tokenizer_type=config.tokenizer_type)
                except:
                    self.tokenizer = SimpleTokenizer.from_pretrained(config.tokenizer_name,tokenizer_type=config.tokenizer_type,add_tab_newline_vocab = config.add_tab_newline_vocab)

        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token if self.tokenizer.eos_token else "[PAD]"
            )

        # Add special tokens
        for token in config.special_tokens.values():
            if token not in self.tokenizer.get_vocab() and config.training_from_scratch:
                self.tokenizer.add_tokens([token])

        # Update config with tokenizer info
        config.vocab_size = len(self.tokenizer)
        config.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        config.eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id

        # Shared components
        self.shared_embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=self.tokenizer.pad_token_id
        )
        nn.init.normal_(self.shared_embedding.weight, mean=0.0, std=0.02)

        # Encoder (shared across all tasks)
        self.encoder = TransformerEncoder(config)
        self.encoder.embedding = self.shared_embedding

        # Task-specific components
        self.decoders = nn.ModuleDict()
        self.task_heads = nn.ModuleDict()

        # Training state
        self.training_step = 0
        self.best_val_loss = float("inf")
        self.steps_without_improvement = 0

        # Continual learning components
        self.ewc_penalties = {}
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.error_mask = None

        # Initialize tasks from config
        if config.tasks:
            for task_name, task_info in config.tasks.items():
                strategy = TrainingStrategy(task_info["strategy"])
                num_labels = task_info.get("num_labels", config.vocab_size)
                self._add_task_head(task_name, strategy, num_labels)
                self.current_tasks[task_name] = task_info

        self.to(config.device)

    def _add_task_head(self, task_name: str, strategy: TrainingStrategy, num_labels: int):
        """Add task-specific head or decoder."""
        if strategy == TrainingStrategy.SENTENCE_CLASSIFICATION:
            head = nn.Sequential(
                nn.Linear(self.config.embed_dim, self.config.embed_dim // 2),
                nn.LayerNorm(self.config.embed_dim // 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.embed_dim // 2, num_labels),
            )
            self.task_heads[task_name] = head.to(self.config.device)

        elif strategy == TrainingStrategy.TOKEN_CLASSIFICATION:
            head = nn.Sequential(
                nn.Linear(self.config.embed_dim, self.config.embed_dim),
                nn.LayerNorm(self.config.embed_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.embed_dim, num_labels),
            )
            self.task_heads[task_name] = head.to(self.config.device)

        elif strategy == TrainingStrategy.CAUSAL_LM:
            decoder = TransformerDecoder(self.config, task_name, has_cross_attention=False)
            decoder.set_shared_embedding(self.shared_embedding)
            self.decoders[task_name] = decoder.to(self.config.device)

        elif strategy == TrainingStrategy.SEQ2SEQ:
            decoder = TransformerDecoder(self.config, task_name, has_cross_attention=True)
            decoder.set_shared_embedding(self.shared_embedding)
            self.decoders[task_name] = decoder.to(self.config.device)

    def add_task(self, task_config: TaskConfig):
        """Add a new task to the model."""
        task_name = task_config.task_name

        if task_name not in self.task_heads and task_name not in self.decoders:
            num_labels = task_config.num_labels or self.config.vocab_size
            self._add_task_head(task_name, task_config.training_strategy, num_labels)

            task_info = {
                "strategy": task_config.training_strategy.value,
                "num_labels": num_labels,
                "best_loss": float("inf"),
                "best_metrics": {},
                "steps_without_improvement": 0,
            }
            self.current_tasks[task_name] = task_info
            self.config.tasks[task_name] = task_info

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_name: Optional[str] = None,
        labels: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        error_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the model.

        Handles different training strategies:
        - SEQ2SEQ: Encoder-decoder with cross-attention
        - CAUSAL_LM: Decoder-only autoregressive
        - SENTENCE_CLASSIFICATION: Classification with pooling
        - TOKEN_CLASSIFICATION: Token-level classification
        """
        if error_mask is not None:
            self.error_mask = error_mask

        if task_name and task_name in self.current_tasks:
            strategy = TrainingStrategy(self.current_tasks[task_name]["strategy"])

            # Sequence-to-Sequence
            if strategy == TrainingStrategy.SEQ2SEQ:
                if encoder_output is None:
                    encoder_output = self.encoder(input_ids, attention_mask, task_name)

                if task_name not in self.decoders:
                    raise ValueError(f"No decoder found for task {task_name}")

                decoder = self.decoders[task_name]

                if labels is not None:
                    decoder_input = labels[:, :-1].clone()
                    decoder_input[decoder_input == -100] = self.tokenizer.pad_token_id or 0
                    target_labels = labels[:, 1:]
                    decoder_mask = (decoder_input != self.tokenizer.pad_token_id).float()

                    logits = decoder(
                        decoder_input, encoder_output, attention_mask, decoder_mask, task_name
                    )

                    loss_fn = nn.CrossEntropyLoss(
                        ignore_index=-100, label_smoothing=self.config.label_smoothing
                    )
                    loss = loss_fn(
                        logits.reshape(-1, self.config.vocab_size), target_labels.reshape(-1)
                    )

                    metrics = {}
                    if not self.training:
                        mask = target_labels != -100
                        if mask.any():
                            preds = logits.argmax(dim=-1)
                            correct = (preds[mask] == target_labels[mask]).float().sum().item()
                            total = mask.sum().item()
                            metrics["token_accuracy"] = correct / total if total > 0 else 0.0

                    return {
                        "logits": logits,
                        "loss": loss,
                        "encoder_output": encoder_output,
                        "metrics": metrics,
                    }
                else:
                    return {"encoder_output": encoder_output}

            # Causal Language Model
            elif strategy == TrainingStrategy.CAUSAL_LM:
                decoder = self.decoders[task_name]

                if labels is not None:
                    decoder_input = labels[:, :-1].clone()
                    decoder_input[decoder_input == -100] = self.tokenizer.pad_token_id or 0
                    target_labels = labels[:, 1:]

                    logits = decoder(decoder_input, task_name=task_name)
                    loss_fn = nn.CrossEntropyLoss(
                        ignore_index=-100, label_smoothing=self.config.label_smoothing
                    )
                    loss = loss_fn(
                        logits.reshape(-1, self.config.vocab_size), target_labels.reshape(-1)
                    )

                    metrics = {}
                    if not self.training:
                        mask = target_labels != -100
                        if mask.any():
                            preds = logits.argmax(dim=-1)
                            metrics["perplexity"] = torch.exp(loss).item()
                            correct = (preds[mask] == target_labels[mask]).float().sum().item()
                            total = mask.sum().item()
                            metrics["token_accuracy"] = correct / total if total > 0 else 0.0

                    return {"logits": logits, "loss": loss, "metrics": metrics}
                else:
                    return {"decoder": decoder}

            # Sentence Classification
            elif strategy == TrainingStrategy.SENTENCE_CLASSIFICATION:
                x = self.shared_embedding(input_ids)
                for layer in self.encoder.layers:
                    x = layer(x, attention_mask, task_name)
                x = self.encoder.norm(x)

                # Pooling
                if attention_mask is not None:
                    mask_sum = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
                else:
                    pooled = x.mean(dim=1)

                logits = self.task_heads[task_name](pooled)

                if labels is not None:
                    if labels.dim() > 1:
                        labels = labels[:, 0]
                    loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
                    loss = loss_fn(logits, labels)

                    metrics = {}
                    if not self.training:
                        preds = logits.argmax(dim=-1)
                        correct = (preds == labels).float().sum().item()
                        total = labels.size(0)
                        metrics["accuracy"] = correct / total if total > 0 else 0.0
                        if total>0:
                            f1_macro, _ = calculate_f_scores(preds, labels, average='macro', beta=1.0)
                            f1_weighted, _ = calculate_f_scores(preds, labels, average='weighted', beta=1.0)
                            f1_micro, _ = calculate_f_scores(preds, labels, average='micro', beta=1.0)
                            metrics["f1_macro"] = f1_macro
                            metrics["f1_weighted"] = f1_weighted
                            metrics["f1_micro"] = f1_micro

                    return {"logits": logits, "loss": loss, "metrics": metrics}
                return {"logits": logits}

            # Token Classification
            elif strategy == TrainingStrategy.TOKEN_CLASSIFICATION:
                x = self.shared_embedding(input_ids)
                for layer in self.encoder.layers:
                    x = layer(x, attention_mask, task_name)
                x = self.encoder.norm(x)

                logits = self.task_heads[task_name](x)

                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss(
                        ignore_index=-100, label_smoothing=self.config.label_smoothing
                    )
                    active = labels.view(-1) != -100

                    if active.any():
                        logits_flat = logits.view(-1, self.task_heads[task_name][-1].out_features)
                        labels_flat = labels.view(-1)
                        loss = loss_fn(logits_flat[active], labels_flat[active])
                    else:
                        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

                    metrics = {}
                    if not self.training and active.any():
                        preds = logits.argmax(dim=-1).view(-1)[active]
                        labels_flat_active = labels_flat[active]

                        correct = (preds == labels_flat_active).float().sum().item()
                        total = active.sum().item()
                        metrics["token_accuracy"] = correct / total if total > 0 else 0.0
                        """
                        is_error = labels_flat_active != 0

                        if is_error.any():
                            error_correct = (preds[is_error] == labels_flat_active[is_error]).float().sum().item()
                            error_total = is_error.sum().item()
                            metrics["token_accuracy(non-zero tags)"] = error_correct / error_total if error_total > 0 else 0.0
                        """
                        if total > 0:
                            # Calculate macro F1 (averaged over all classes)
                            f1_macro, per_class_f1 = calculate_f_scores(preds, labels_flat_active, average='macro', beta=1.0)
                            
                            #f05_macro, per_class_f05 = calculate_f_scores(preds, labels_flat_active, average='macro', beta=0.5)
                            
                            # Calculate weighted F1 (accounting for class imbalance)
                            f1_weighted, _ = calculate_f_scores(preds, labels_flat_active, average='weighted', beta=1.0)
                            #f05_weighted, _ = calculate_f_scores(preds, labels_flat_active, average='weighted', beta=0.5)
                            
                            # Calculate micro F1 (overall performance)
                            f1_micro, _ = calculate_f_scores(preds, labels_flat_active, average='micro', beta=1.0)
                            #f05_micro, _ = calculate_f_scores(preds, labels_flat_active, average='micro', beta=0.5)
                            
                            metrics["f1_macro"] = f1_macro
                            #metrics["f0.5_macro"] = f05_macro
                            metrics["f1_weighted"] = f1_weighted
                            #metrics["f0.5_weighted"] = f05_weighted
                            metrics["f1_micro"] = f1_micro
                            #metrics["f0.5_micro"] = f05_micro
                            
                            # Optional: Add per-class scores (useful for debugging)
                            # metrics["per_class_f1"] = {str(k): v for k, v in per_class_f1.items()}
                            

                    return {"logits": logits, "loss": loss, "metrics": metrics}
                return {"logits": logits}

        # Fallback: just encode
        x = self.shared_embedding(input_ids)
        for layer in self.encoder.layers:
            x = layer(x, attention_mask, None)
        x = self.encoder.norm(x)
        return {"hidden_states": x}

    def generate(
        self,
        prompt: str,
        task_name: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        decoding_with_special_tokens: str = "no",
    ):
        """Generate text or predictions for a given prompt."""
        self.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self.config.device)
        attention_mask = inputs["attention_mask"].to(self.config.device)

        if task_name and task_name in self.current_tasks:
            strategy = TrainingStrategy(self.current_tasks[task_name]["strategy"])

            if strategy == TrainingStrategy.CAUSAL_LM:
                decoder = self.decoders[task_name]
                with torch.no_grad():
                    generated_ids = decoder.generate(
                        input_ids=input_ids,
                        max_len=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                if decoding_with_special_tokens == "no":
                    return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                elif decoding_with_special_tokens == "yes":
                    return self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
                else:
                    return generated_ids

            elif strategy == TrainingStrategy.SEQ2SEQ:
                with torch.no_grad():
                    encoder_output = self.encoder(input_ids, attention_mask, task_name)
                    decoder = self.decoders[task_name]
                    generated_ids = decoder.generate(
                        encoder_output=encoder_output,
                        src_mask=attention_mask,
                        max_len=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                if decoding_with_special_tokens == "no":
                    return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                elif decoding_with_special_tokens == "yes":
                    return self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
                else:
                    return generated_ids

            elif strategy == TrainingStrategy.SENTENCE_CLASSIFICATION:
                with torch.no_grad():
                    outputs = self.forward(input_ids, attention_mask, task_name=task_name)
                    logits = outputs["logits"]
                    probs = F.softmax(logits, dim=-1)
                    pred_id = probs.argmax().item()
                    return {
                        "prediction": pred_id,
                        "probabilities": probs[0].cpu().tolist(),
                        "logits": logits[0].cpu().tolist(),
                    }

            elif strategy == TrainingStrategy.TOKEN_CLASSIFICATION:
                with torch.no_grad():
                    outputs = self.forward(input_ids, attention_mask, task_name=task_name)
                    logits = outputs["logits"]
                    predictions = logits.argmax(dim=-1)[0].cpu().tolist()
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                    return {
                        "tokens": tokens,
                        "predictions": predictions,
                        "pairs": list(zip(tokens, predictions)),
                    }

        return prompt

    def save_model(self, path: str):
        """Save the entire model to disk."""
        self.config.tasks = self.current_tasks
        state = {
            "model_state": self.state_dict(),
            "config": self.config,
            "current_tasks": self.current_tasks,
            "training_step": self.training_step,
            "best_val_loss": self.best_val_loss,
            "steps_without_improvement": self.steps_without_improvement,
            "tokenizer_name": self.config.tokenizer_name,
            "special_tokens": self.config.special_tokens,
        }
        torch.save(state, path)

    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None):
        """Load a saved model from disk."""
        try:
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            state = torch.load(path, map_location=device, weights_only=False)
            config = state["config"]
            config.device = device

            model = cls(config)
            model.load_state_dict(state["model_state"], strict=False)
            model.current_tasks = state.get("current_tasks", {})
            model.training_step = state.get("training_step", 0)
            model.best_val_loss = state.get("best_val_loss", float("inf"))
            model.steps_without_improvement = state.get("steps_without_improvement", 0)

            return model.to(config.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_exporter(self):
        """Get a model exporter instance."""
        return ModelExporter(self)
