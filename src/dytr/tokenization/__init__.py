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
Tokenization module for Dynamic Transformers.

This module provides tokenizer implementations and download utilities.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""


from dytr.tokenization.download_manager import (
    DownloadManager,
    download_file,
    download_tokenizer_files,
    get_url_from_HF,
    process_vocab_text,
)
from dytr.tokenization.simple_tokenizer import SimpleTokenizer

__all__ = [
    "SimpleTokenizer",
    "DownloadManager",
    "download_file",
    "download_tokenizer_files",
    "get_url_from_HF",
    "process_vocab_text",
]
