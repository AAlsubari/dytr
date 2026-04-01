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
Experience replay buffer for continual learning.

This module implements a replay buffer that stores samples from previous tasks
to prevent catastrophic forgetting.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""

import random
from collections import defaultdict
from typing import Any, List, Optional


class ReplayBuffer:
    """
    Experience replay buffer for storing samples from previous tasks.

    The buffer maintains a fixed capacity and stores samples along with
    their task labels for later replay during training.

    Args:
        capacity: Maximum number of samples to store
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = []
        self.task_indices = defaultdict(list)

    def add_samples(self, task_name: str, samples: List[Any]):
        """
        Add samples to the replay buffer.

        Args:
            task_name: Name of the task these samples belong to
            samples: List of samples to add
        """
        for sample in samples:
            # Remove oldest sample if buffer is full
            if len(self.buffer) >= self.capacity:
                oldest_task = self.buffer[0][0]
                self.task_indices[oldest_task].pop(0)
                self.buffer.pop(0)

            self.buffer.append((task_name, sample))
            self.task_indices[task_name].append(len(self.buffer) - 1)

    def sample(self, batch_size: int, task_name: Optional[str] = None) -> List[Any]:
        """
        Sample random samples from the buffer.

        Args:
            batch_size: Number of samples to retrieve
            task_name: If specified, only sample from this task

        Returns:
            List of sampled samples
        """
        if task_name and task_name in self.task_indices and self.task_indices[task_name]:
            indices = random.sample(
                self.task_indices[task_name], min(batch_size, len(self.task_indices[task_name]))
            )
            return [self.buffer[i][1] for i in indices]
        elif self.buffer:
            indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
            return [self.buffer[i][1] for i in indices]#, [self.buffer[i][0] for i in indices]
        return []
