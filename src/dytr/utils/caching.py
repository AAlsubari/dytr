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
Caching utilities for dataset and model caching.

This module provides helper functions for managing cache directories
and file operations.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""
__version__ = "0.1.0"
__author__ = "Akram Alsubari"
__email__ = "akram.alsubari@outlook.com"
__license__ = "Apache 2.0"
#import os
from pathlib import Path
from typing import Optional


class CacheManager:
    """
    Manages cache directories for datasets and models.

    Args:
        cache_dir: Root cache directory path
    """

    def __init__(self, cache_dir: str = "./dataset_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_dataset_cache_path(
        self, dataset_name: str, task_name: str, max_len: int, stride: int
    ) -> Path:
        """Get cache path for a processed dataset."""
        cache_subdir = self.cache_dir / "datasets" / task_name
        cache_subdir.mkdir(parents=True, exist_ok=True)
        filename = f"{dataset_name}_{max_len}_{stride}.pkl"
        return cache_subdir / filename

    def get_model_cache_path(self, model_name: str, task_name: str) -> Path:
        """Get cache path for a model."""
        cache_subdir = self.cache_dir / "models" / task_name
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / f"{model_name}.pt"

    def get_tokenizer_cache_path(self, tokenizer_name: str) -> Path:
        """Get cache path for a tokenizer."""
        cache_subdir = self.cache_dir / "tokenizers"
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / tokenizer_name

    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache files."""
        if cache_type == "datasets":
            cache_path = self.cache_dir / "datasets"
        elif cache_type == "models":
            cache_path = self.cache_dir / "models"
        elif cache_type == "tokenizers":
            cache_path = self.cache_dir / "tokenizers"
        else:
            cache_path = self.cache_dir

        if cache_path.exists():
            import shutil

            shutil.rmtree(cache_path)
            print(f"Cleared cache at {cache_path}")

    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size


# Global cache manager instance
_default_cache_manager = None


def get_cache_manager(cache_dir: str = "./dataset_cache") -> CacheManager:
    """Get or create a cache manager instance."""
    global _default_cache_manager
    if _default_cache_manager is None or _default_cache_manager.cache_dir != Path(cache_dir):
        _default_cache_manager = CacheManager(cache_dir)
    return _default_cache_manager
