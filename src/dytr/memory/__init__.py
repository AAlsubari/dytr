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
Memory and continual learning modules for Dynamic Transformers.

This module provides implementations of continual learning techniques
including Elastic Weight Consolidation (EWC) and Experience Replay
to prevent catastrophic forgetting when training on sequential tasks.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""

from dytr.memory.ewc import EWC
from dytr.memory.replay import ReplayBuffer

__all__ = ["EWC", "ReplayBuffer"]
