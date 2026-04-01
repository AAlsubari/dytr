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
Download manager for tokenizer files from HuggingFace.

This module provides functionality to download tokenizer files from HuggingFace
with resume support and caching.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""


import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import requests
from tqdm import tqdm


def get_url_from_HF(HF_path: str):
    """
    Generate download URLs for HuggingFace tokenizer files.

    Args:
        HF_path: HuggingFace model path (e.g., 'bert-base-uncased')

    Returns:
        Tuple of (directory_name, vocab_url, special_tokens_map_url)
    """
    vocab_url = f"https://huggingface.co/{HF_path}/resolve/main/vocab.txt?download=true"
    special_token_map = (
        f"https://huggingface.co/{HF_path}/resolve/main/special_tokens_map.json?download=true"
    )
    file_dir = HF_path.split("/")[-1].replace("-", "_")
    return file_dir, vocab_url, special_token_map


def process_vocab_text(path: Union[str, Path]) -> Dict[str, int]:
    """
    Process vocabulary text file into token-to-id mapping.

    Args:
        path: Path to vocabulary text file

    Returns:
        Dictionary mapping tokens to IDs
    """
    path = Path(path)
    json_path = path.with_suffix(".json")

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        print("JSON OF VOCAB ALREADY SAVED")
        return token_to_id

    print("*" * 60)
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    token_to_id = {k: idx for idx, k in enumerate(data.split("\n")) if k}
    return token_to_id


class DownloadManager:
    """
    Manages file downloads with resume support and caching.

    This class handles downloading files from URLs, supports resuming
    interrupted downloads, and caches downloaded files to avoid
    re-downloading.

    Args:
        base_dir: Base directory for storing downloads
        registry_file: Name of the registry file for tracking downloads
    """

    def __init__(self, base_dir: str = "downloads", registry_file: str = "download_registry.pkl"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.registry_file = self.base_dir / registry_file
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, str]:
        """Load download registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_registry(self):
        """Save download registry to disk."""
        with open(self.registry_file, "wb") as f:
            pickle.dump(self.registry, f)

    def _get_file_path(self, url: str, subfolder: Optional[str] = None) -> Path:
        """
        Generate file path for a given URL.

        Args:
            url: Download URL
            subfolder: Optional subfolder within base directory

        Returns:
            Path object for the file
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        original_filename = url.split("/")[-1].split("?")[0]

        if not original_filename:
            original_filename = f"download_{url_hash}"

        if subfolder:
            file_path = self.base_dir / subfolder / original_filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path = self.base_dir / original_filename

        return file_path

    def download_file(
        self, url: str, subfolder: Optional[str] = None
    ) -> Union[str, pd.DataFrame, dict, list]:
        """
        Download a file from URL with resume support.

        Args:
            url: URL to download
            subfolder: Optional subfolder to store the file

        Returns:
            Parsed file content (dict for JSON, DataFrame for CSV/Excel, str for text)
        """
        file_path = self._get_file_path(url, subfolder)

        # Check if file already exists in registry
        if str(file_path) in self.registry.values():
            if file_path.exists():
                print(f"File already exists: {file_path}")
                return self._read_file(file_path)

        # Check if file exists directly
        if file_path.exists():
            print(f"File already exists: {file_path}")
            return self._read_file(file_path)

        # Download file
        self._download_with_resume(url, file_path)

        # Update registry
        self.registry[url] = str(file_path)
        self._save_registry()

        return self._read_file(file_path)

    def _download_with_resume(self, url: str, file_path: Path):
        """
        Download file with resume support for interrupted downloads.

        Args:
            url: URL to download
            file_path: Path where to save the file
        """
        existing_size = file_path.stat().st_size if file_path.exists() else 0

        headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}

        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0)) + existing_size

        mode = "ab" if existing_size > 0 else "wb"

        with open(file_path, mode) as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {file_path.name}",
                initial=existing_size,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    def _read_file(self, file_path: Path) -> Union[str, pd.DataFrame, dict, list]:
        """
        Read and parse a downloaded file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Parsed file content
        """
        file_extension = file_path.suffix.lower()

        if file_extension == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif file_extension == ".csv":
            return pd.read_csv(file_path)
        elif file_extension == ".txt":
            if "vocab" in str(file_path).lower():
                return process_vocab_text(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_extension in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    def get_downloaded_files(self) -> Dict[str, str]:
        """
        Get all downloaded files from registry.

        Returns:
            Dictionary mapping URLs to file paths
        """
        return self.registry.copy()


def download_file(
    url: str, subfolder: Optional[str] = None, base_dir: str = "downloads"
) -> Union[str, pd.DataFrame, dict, list]:
    """
    Convenience function to download a single file.

    Args:
        url: URL to download
        subfolder: Optional subfolder for the file
        base_dir: Base directory for downloads

    Returns:
        Parsed file content
    """
    manager = DownloadManager(base_dir)
    return manager.download_file(url, subfolder)


def download_tokenizer_files(HF_tokenizer_path: str) -> tuple:
    """
    Download tokenizer files from HuggingFace.

    Args:
        HF_tokenizer_path: HuggingFace tokenizer path (e.g., 'bert-base-uncased')

    Returns:
        Tuple of (vocab_dict, special_tokens_map)
    """
    sub_dir, vocab_path, special_token_map_path = get_url_from_HF(HF_tokenizer_path)
    vocab = download_file(vocab_path, sub_dir)
    try:
        special_token_map = download_file(special_token_map_path, sub_dir)
    except:
        special_token_map = {}

    return vocab, special_token_map
