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
Model exporter for extracting single-task models from multi-task DynamicTransformer.

This module provides functionality to export individual task models for inference,
supporting all training strategies.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dytr.core.config import TrainingStrategy


class ModelExporter:
    """
    Exports single-task models from a multi-task DynamicTransformer.

    This class allows extracting individual task models for deployment,
    reducing memory footprint and simplifying inference.

    Args:
        model: The DynamicTransformer instance to export from
    """

    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        self.config = model.config

    def export_single_task(self, task_name: str, save_path: str):
        """
        Export a single task model to a file.

        Args:
            task_name: Name of the task to export
            save_path: Path where to save the exported model
        """
        if task_name not in self.model.current_tasks:
            raise ValueError(f"Task {task_name} not found in model")

        strategy = TrainingStrategy(self.model.current_tasks[task_name]["strategy"])
        num_labels = self.model.current_tasks[task_name].get("num_labels")

        # Sentence Classification Task
        if strategy == TrainingStrategy.SENTENCE_CLASSIFICATION:
            if num_labels is None:
                task_head = self.model.task_heads[task_name]
                num_labels = (
                    task_head[-1].out_features
                    if isinstance(task_head, nn.Sequential)
                    else task_head.out_features
                )

            class SingleTaskModel(nn.Module):
                def __init__(self, encoder, task_head, config, tokenizer, num_labels, task_name):
                    super().__init__()
                    self.encoder = encoder
                    self.task_head = task_head
                    self.config = config
                    self.tokenizer = tokenizer
                    self.num_labels = num_labels
                    self.task_name = task_name
                    self.strategy = "sentence_classification"

                def forward(self, input_ids, attention_mask=None):
                    x = self.encoder(input_ids, attention_mask, self.task_name)
                    if attention_mask is not None:
                        mask_sum = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                        pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
                    else:
                        pooled = x.mean(dim=1)
                    logits = self.task_head(pooled)
                    return logits

                def generate(self, text, **kwargs):
                    self.eval()
                    inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        logits = self.forward(inputs["input_ids"], inputs.get("attention_mask"))
                        probs = F.softmax(logits, dim=-1)
                        pred_id = probs.argmax().item()
                        return {
                            "text": text,
                            "prediction": pred_id,
                            "probabilities": probs[0].cpu().tolist(),
                            "logits": logits[0].cpu().tolist(),
                            "strategy": self.strategy,
                        }

            model = SingleTaskModel(
                self.model.encoder,
                self.model.task_heads[task_name],
                self.config,
                self.tokenizer,
                num_labels,
                task_name,
            )

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": self.config,
                    "task_name": task_name,
                    "strategy": strategy.value,
                    "tokenizer_name": self.config.tokenizer_name,
                    "num_labels": num_labels,
                },
                save_path,
            )
            return model

        # Token Classification Task
        elif strategy == TrainingStrategy.TOKEN_CLASSIFICATION:
            if num_labels is None:
                task_head = self.model.task_heads[task_name]
                num_labels = (
                    task_head[-1].out_features
                    if isinstance(task_head, nn.Sequential)
                    else task_head.out_features
                )

            class SingleTaskModel(nn.Module):
                def __init__(self, encoder, task_head, config, tokenizer, num_labels, task_name):
                    super().__init__()
                    self.encoder = encoder
                    self.task_head = task_head
                    self.config = config
                    self.tokenizer = tokenizer
                    self.num_labels = num_labels
                    self.task_name = task_name
                    self.strategy = "token_classification"

                def forward(self, input_ids, attention_mask=None):
                    x = self.encoder(input_ids, attention_mask, self.task_name)
                    logits = self.task_head(x)
                    return logits

                def generate(self, text, **kwargs):
                    self.eval()
                    inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        logits = self.forward(inputs["input_ids"], inputs.get("attention_mask"))
                        predictions = logits.argmax(dim=-1)[0].cpu().tolist()
                        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                        return {
                            "text": text,
                            "tokens": tokens,
                            "predictions": predictions,
                            "pairs": list(zip(tokens, predictions)),
                            "strategy": self.strategy,
                        }

            model = SingleTaskModel(
                self.model.encoder,
                self.model.task_heads[task_name],
                self.config,
                self.tokenizer,
                num_labels,
                task_name,
            )

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": self.config,
                    "task_name": task_name,
                    "strategy": strategy.value,
                    "tokenizer_name": self.config.tokenizer_name,
                    "num_labels": num_labels,
                },
                save_path,
            )
            return model

        # Seq2Seq Task
        elif strategy == TrainingStrategy.SEQ2SEQ:

            class Seq2SeqModel(nn.Module):
                def __init__(self, encoder, decoder, config, tokenizer, task_name):
                    super().__init__()
                    self.encoder = encoder
                    self.decoder = decoder
                    self.config = config
                    self.tokenizer = tokenizer
                    self.task_name = task_name
                    self.strategy = "seq2seq"

                def forward(self, input_ids, attention_mask=None, labels=None):
                    encoder_output = self.encoder(input_ids, attention_mask, self.task_name)
                    if labels is not None:
                        decoder_input = labels[:, :-1]
                        logits = self.decoder(decoder_input, encoder_output, attention_mask)
                        return logits
                    return encoder_output

                def generate(
                    self,
                    text,
                    max_length=50,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.1,
                    **kwargs,
                ):
                    self.eval()
                    inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        encoder_output = self.encoder(
                            inputs["input_ids"], inputs["attention_mask"], self.task_name
                        )
                        generated_ids = self.decoder.generate(
                            encoder_output=encoder_output,
                            src_mask=inputs["attention_mask"],
                            max_len=max_length,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                        )
                        generated_text = self.tokenizer.decode(
                            generated_ids[0], skip_special_tokens=True
                        )
                        return {
                            "source": text,
                            "generated": generated_text,
                            "strategy": self.strategy,
                        }

            model = Seq2SeqModel(
                self.model.encoder,
                self.model.decoders[task_name],
                self.config,
                self.tokenizer,
                task_name,
            )

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": self.config,
                    "task_name": task_name,
                    "strategy": strategy.value,
                    "tokenizer_name": self.config.tokenizer_name,
                },
                save_path,
            )
            return model

        # Causal LM Task
        elif strategy == TrainingStrategy.CAUSAL_LM:

            class DecoderOnlyModel(nn.Module):
                def __init__(self, decoder, config, tokenizer, task_name):
                    super().__init__()
                    self.decoder = decoder
                    self.config = config
                    self.tokenizer = tokenizer
                    self.task_name = task_name
                    self.strategy = "causal_lm"

                def forward(self, input_ids, attention_mask=None):
                    return self.decoder(input_ids)

                def generate(
                    self,
                    prompt,
                    max_length=50,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.1,
                    **kwargs,
                ):
                    self.eval()
                    inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        generated_ids = self.decoder.generate(
                            input_ids=inputs["input_ids"],
                            max_len=max_length,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                        )
                        generated_text = self.tokenizer.decode(
                            generated_ids[0], skip_special_tokens=True
                        )
                        return {
                            "prompt": prompt,
                            "generated": generated_text,
                            "strategy": self.strategy,
                        }

            model = DecoderOnlyModel(
                self.model.decoders[task_name], self.config, self.tokenizer, task_name
            )

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": self.config,
                    "task_name": task_name,
                    "strategy": strategy.value,
                    "tokenizer_name": self.config.tokenizer_name,
                },
                save_path,
            )
            return model

    def export_all_tasks(self, save_dir: str) -> Dict[str, nn.Module]:
        """
        Export all tasks from the model.

        Args:
            save_dir: Directory where to save exported models

        Returns:
            Dictionary mapping task names to exported models
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        exported_models = {}
        for task_name in self.model.current_tasks.keys():
            save_path = save_dir / f"{task_name}.pt"
            print(f"  Exporting {task_name} -> {save_path}")
            model = self.export_single_task(task_name, str(save_path))
            exported_models[task_name] = model

        return exported_models
