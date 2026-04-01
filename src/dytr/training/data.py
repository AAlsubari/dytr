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
Data utilities for multi-task training.

This module provides dataset classes and samplers for handling multiple tasks
during training.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""



import bisect
from collections import defaultdict
from typing import  Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def collate_fn(batch, fixed_max_len: Optional[int] = None):
    """
    Custom collate function for multi-task batches.

    Handles variable-length sequences and different task types.

    Args:
        batch: List of samples from the dataset

    Returns:
        Dictionary with batched tensors
    """
    if not isinstance(batch, list):
        batch = [batch]

    batch_dict = {}

    for key in batch[0].keys():
        if key in ["input_ids", "attention_mask", "labels"]:
            values = []
            for item in batch:
                if key in item and isinstance(item[key], torch.Tensor):
                    if item[key].dim() > 0:
                        values.append(item[key])

            if not values:
                continue

            # Find max length for padding
            if fixed_max_len is not None:
                # Use fixed length for all sequences
                max_len = batch[0]["max_length"]  # fixed_max_len 
            else:
                # Dynamic: use max length in this batch
                max_len = max(v.size(0) for v in values if v.dim() > 0)
            # max_len = max(v.size(0) for v in values if v.dim() > 0)
            padded_values = []

            for v in values:
                if v.dim() > 0:
                    if v.size(0) < max_len:
                        pad_size = max_len - v.size(0)
                        if key == "labels":
                            padded = F.pad(v, (0, pad_size), value=-100)
                        else:
                            padded = F.pad(v, (0, pad_size), value=0)
                    else:
                        padded = v
                    padded_values.append(padded.unsqueeze(0))

            if padded_values:
                batch_dict[key] = torch.cat(padded_values, dim=0)
        else:
            # For non-tensor fields, keep as list
            batch_dict[key] = [item[key] for item in batch if key in item]

    return batch_dict


class TaskAwareBatchSampler:
    """
    Batch sampler that creates batches with samples from the same task.

    Ensures that each batch contains samples from only one task to simplify
    task-specific processing.

    Args:
        dataset: MultiTaskDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle batches
        drop_last: Whether to drop incomplete batches
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Group indices by task
        self.indices_by_task = defaultdict(list)
        for idx in range(len(dataset)):
            dataset_idx = bisect.bisect_right(dataset.cumulative_sizes, idx) - 1
            task_name = dataset.task_names[dataset_idx]
            self.indices_by_task[task_name].append(idx)

        # Create batches
        self.batches = []
        self.batch_task_map = []

        for task_name, indices in self.indices_by_task.items():
            if shuffle:
                np.random.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                if drop_last and len(batch_indices) < batch_size:
                    continue
                self.batches.append(batch_indices)
                self.batch_task_map.append(task_name)

        # Shuffle batches if needed
        if shuffle:
            combined = list(zip(self.batches, self.batch_task_map))
            np.random.shuffle(combined)
            if combined:
                self.batches, self.batch_task_map = zip(*combined)
                self.batches = list(self.batches)
                self.batch_task_map = list(self.batch_task_map)

    def __iter__(self):
        """Iterate over batches."""
        for batch_indices in self.batches:
            yield batch_indices

    def __len__(self):
        """Number of batches."""
        return len(self.batches)


class MultiTaskDataset(Dataset):
    """
    Dataset that combines multiple single-task datasets.

    Creates a unified dataset from multiple task-specific datasets,
    allowing seamless multi-task training.

    Args:
        datasets_dict: Dictionary mapping task names to (dataset, strategy) tuples
        tokenizer: Tokenizer instance
        task_configs: List of task configurations
    """

    def __init__(self, datasets_dict, tokenizer, task_configs):
        self.datasets = []
        self.task_names = []
        self.max_lengths = {t.task_name: t.max_length for t in task_configs}
        self.strategies = []
        self.cumulative_sizes = [0]

        print("Max lengths per task:", self.max_lengths)

        for task_name, (dataset, strategy) in datasets_dict.items():
            if len(dataset) > 0:
                self.datasets.append(dataset)
                self.task_names.append(task_name)
                self.strategies.append(strategy)
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(dataset))

    def __len__(self):
        """Total number of samples across all tasks."""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        """
        Get a sample by index.

        Args:
            idx: Index in the combined dataset

        Returns:
            Sample with task_name and strategy added
        """
        if idx < 0:
            idx = len(self) + idx

        # Find which dataset this index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        sample_idx = idx - self.cumulative_sizes[dataset_idx]

        # Get the sample from the specific dataset
        item = self.datasets[dataset_idx][sample_idx]

        if isinstance(item, dict):
            item["task_name"] = self.task_names[dataset_idx]
            item["strategy"] = self.strategies[dataset_idx].value
            item["max_length"] = self.max_lengths[self.task_names[dataset_idx]]

        return item
