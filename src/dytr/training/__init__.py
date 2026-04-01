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
Training modules for Dynamic Transformers.

This module provides the trainer class, dataset handling utilities,
and data processing functions for multi-task training with continual learning support.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""



from dytr.training.data import MultiTaskDataset, TaskAwareBatchSampler, collate_fn
from dytr.training.dataset import SingleDatasetProcessing
from dytr.training.trainer import Trainer
from dytr.training.utils import process_qa_dataset, set_seed

__all__ = [
    "Trainer",
    "SingleDatasetProcessing",
    "MultiTaskDataset",
    "TaskAwareBatchSampler",
    "collate_fn",
    "set_seed",
    "process_qa_dataset",
]

