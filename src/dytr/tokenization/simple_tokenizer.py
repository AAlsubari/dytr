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
Simple tokenizer implementation with multiple tokenization strategies.

This module provides a lightweight tokenizer that supports various tokenization
methods including WordPiece, BPE, Unigram, and Byte-Level encoding.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""


import json
import os
from typing import Any, Dict, List, Optional, Union


class SimpleTokenizer:
    """
    A flexible tokenizer supporting multiple tokenization strategies.

    Supports:
    - WordPiece tokenization
    - Byte-Pair Encoding (BPE)
    - Unigram tokenization
    - Character-level tokenization
    - Space-based tokenization
    - Byte-level tokenization (GPT-2 style)

    Args:
        vocab: Vocabulary dictionary mapping tokens to IDs
        special_tokens_map: Dictionary of special tokens
        tokenizer_type: Type of tokenizer ('wordpiece', 'bpe', 'unigram', 'char', 'space', 'byte_level')
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        special_tokens_map: Optional[Dict] = None,
        tokenizer_type: str = "wordpiece", add_tab_newline_vocab: bool =False
    ):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.special_tokens_map = special_tokens_map or {}
        self.tokenizer_type = tokenizer_type
        self.add_tab_newline_vocab = add_tab_newline_vocab

        # Special tokens
        self.pad_token = self.special_tokens_map.get("pad_token", "[PAD]")
        self.unk_token = self.special_tokens_map.get("unk_token", "[UNK]")
        self.cls_token = self.special_tokens_map.get("cls_token", "[CLS]")
        self.sep_token = self.special_tokens_map.get("sep_token", "[SEP]")
        self.mask_token = self.special_tokens_map.get("mask_token", "[MASK]")
        self.bos_token = self.special_tokens_map.get("bos_token", "<s>")
        self.eos_token = self.special_tokens_map.get("eos_token", "</s>")

        # Token IDs
        self.pad_token_id = self.vocab.get(self.pad_token)
        self.unk_token_id = self.vocab.get(self.unk_token)
        self.cls_token_id = self.vocab.get(self.cls_token)
        self.sep_token_id = self.vocab.get(self.sep_token)
        self.mask_token_id = self.vocab.get(self.mask_token)

        # Add BOS and EOS if not present
        if self.bos_token not in self.vocab:
            self.bos_token=self.cls_token
            
            #self.add_tokens([self.bos_token])
        if self.eos_token not in self.vocab:
            self.eos_token=self.sep_token
            #self.add_tokens([self.eos_token])

        self.bos_token_id = self.vocab.get(self.bos_token)
        self.eos_token_id = self.vocab.get(self.eos_token)

        # All special tokens
        self.all_special_tokens = [
            self.pad_token,
            self.unk_token,
            self.cls_token,
            self.sep_token,
            self.mask_token,
            self.bos_token,
            self.eos_token,
        ]
        self.all_special_ids = [
            self.pad_token_id,
            self.unk_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
            self.bos_token_id,
            self.eos_token_id,
        ]
        
        if self.add_tab_newline_vocab:
            # Add newline and tab representations
            self.add_tokens(["ĉ", "Ċ","Ġ"])
        self.name_or_path = "local_tokenizer"

        # Tokenization function mapping
        self.tokenization_functions = {
            "wordpiece": self._wordpiece_tokenize,
            "bpe": self._bpe_tokenize,
            "unigram": self._unigram_tokenize,
            "char": self._char_tokenize,
            "space": self._space_tokenize,
            "byte_level": self._byte_level_tokenize,
        }

        # BPE merges
        self.merges = self.special_tokens_map.get("merges", [])

        # Byte-level encoder
        self.byte_encoder = self._build_byte_encoder() if tokenizer_type == "byte_level" else None

    def _build_byte_encoder(self) -> Dict[str, str]:
        """Build byte-level encoder for GPT-2 style tokenization."""
        # Printable ASCII range
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(ord("¡"), ord("¬") + 1))
        bs += list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return {chr(b): chr(c) for b, c in zip(bs, cs)}

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using the configured tokenization strategy."""
        if not text:
            return []

        tokenizer_func = self.tokenization_functions.get(
            self.tokenizer_type, self._wordpiece_tokenize
        )
        return tokenizer_func(text)

    def _space_tokenize(self, text: str) -> List[str]:
        """Simple space-based tokenization."""
        return text.split()

    def _char_tokenize(self, text: str) -> List[str]:
        """Character-level tokenization."""
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(char)
            else:
                tokens.append(self.unk_token)
        return tokens

    def _wordpiece_tokenize(self, text: str) -> List[str]:
        """WordPiece tokenization."""
        if self.add_tab_newline_vocab:
            text = text.replace("\n", "Ċ").replace("    ", "\t").replace("\t", "ĉ")
        words = text.split(" ")
        tokens = []

        for word in words:
            if word in self.vocab:
                tokens.append(word)
                continue
            subwords = self._wordpiece_split(word)
            tokens.extend(subwords)

        return tokens

    def _wordpiece_split(self, word: str) -> List[str]:
        """Split a word into WordPiece subwords."""
        if not word:
            return []
        if word in self.vocab:
            return [word]

        tokens = []
        remaining = word
        start = 0

        while start < len(remaining):
            end = len(remaining)
            best_token = None

            while start < end:
                subword = remaining[start:end]
                if start > 0:
                    subword = "##" + subword
                if subword in self.vocab:
                    best_token = subword
                    break
                end -= 1

            if best_token is None:
                # Unknown characters
                for char in remaining[start:]:
                    if char in self.vocab:
                        tokens.append(char)
                    else:
                        tokens.append(self.unk_token)
                break
            else:
                tokens.append(best_token)
                start = end

        return tokens

    def _bpe_tokenize(self, text: str) -> List[str]:
        """Byte-Pair Encoding tokenization."""
        words = self._preprocess_text(text)
        tokens = []

        for word in words:
            if word in self.vocab:
                tokens.append(word)
            else:
                bpe_tokens = self._apply_bpe(word)
                tokens.extend(bpe_tokens)

        return tokens

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BPE tokenization."""
        processed = []
        for i, char in enumerate(text):
            if char == " ":
                processed.append("Ġ")
            elif char == "\n":
                count = 1
                idx = i + 1
                while idx < len(text) and text[idx] == "\n":
                    count += 1
                    idx += 1
                processed.append("Ċ" * count)
            elif char == "\t":
                processed.append("ĉ")
            else:
                processed.append(char)
        return processed

    def _apply_bpe(self, word: Union[str, List[str]]) -> List[str]:
        """Apply BPE merges to a word."""
        if isinstance(word, list):
            word = "".join(word)

        if not self.merges:
            return [word] if word in self.vocab else [self.unk_token]

        tokens = list(word)

        for merge in self.merges:
            pair = merge.split()
            if len(pair) != 2:
                continue

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append("".join(pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [token if token in self.vocab else self.unk_token for token in tokens]

    def _unigram_tokenize(self, text: str) -> List[str]:
        """Unigram tokenization using Viterbi algorithm."""
        chars = list(text)
        n = len(chars)
        dp = [-1] * (n + 1)
        dp[0] = 0

        # Find best segmentation
        for i in range(n):
            if dp[i] == -1:
                continue
            for j in range(i + 1, min(i + 20, n + 1)):
                token = "".join(chars[i:j])
                if token in self.vocab or token in self.all_special_tokens:
                    if dp[j] == -1 or dp[j] > dp[i] + 1:
                        dp[j] = dp[i] + 1

        if dp[n] == -1:
            return [self.unk_token]

        # Reconstruct tokens
        tokens = []
        i = n
        while i > 0:
            for j in range(max(0, i - 20), i):
                if dp[j] != -1 and dp[i] == dp[j] + 1:
                    token = "".join(chars[j:i])
                    if token in self.vocab:
                        tokens.insert(0, token)
                    else:
                        tokens.insert(0, self.unk_token)
                    i = j
                    break

        return tokens

    def _byte_level_tokenize(self, text: str) -> List[str]:
        """Byte-level tokenization (GPT-2 style)."""
        if not self.byte_encoder:
            return self._char_tokenize(text)

        # Encode bytes
        bytes_text = text.encode("utf-8")
        encoded_chars = []
        for b in bytes_text:
            if b < 128:
                encoded_chars.append(chr(b))
            else:
                encoded_chars.append(self.byte_encoder.get(chr(b), chr(b)))

        # Preprocess
        processed = []
        i = 0
        while i < len(encoded_chars):
            if encoded_chars[i] == " ":
                processed.append("Ġ")
                i += 1
            elif encoded_chars[i] == "\n":
                count = 1
                while i + count < len(encoded_chars) and encoded_chars[i + count] == "\n":
                    count += 1
                processed.append("Ċ" * count)
                i += count
            elif encoded_chars[i] == "\t":
                processed.append("ĉ")
                i += 1
            else:
                processed.append(encoded_chars[i])
                i += 1

        # Apply BPE if merges exist
        if self.merges:
            return self._apply_bpe(processed)

        return [p if p in self.vocab else self.unk_token for p in processed]

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> List[int]:
        """Convert tokens to token IDs."""
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> List[str]:
        """Convert token IDs to tokens."""
        if hasattr(ids, "cpu"):
            ids = ids.cpu().tolist()
        elif isinstance(ids, int):
            ids = [ids]

        tokens = []
        for idx in ids:
            token = self.inv_vocab.get(idx, self.unk_token)
            if skip_special_tokens and token in self.all_special_tokens:
                continue
            tokens.append(token)

        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a list of tokens to a string."""
        text = ""
        for i, token in enumerate(tokens):
            if token.startswith("##"):
                text += token[2:]
            elif token in ["Ġ", "Ċ", "ĉ"] or (
                self.tokenizer_type == "byte_level" and token in ["Ġ", "Ċ", "ĉ"]
            ):
                if token == "Ġ":
                    text += " "
                elif token == "Ċ":
                    text += "\n"
                elif token == "ĉ":
                    text += "\t"
            elif token in self.all_special_tokens:
                if token == self.sep_token and i == len(tokens) - 1:
                    continue
                text += " " + token
            else:
                if (
                    text
                    and not text.endswith(" ")
                    and not text.endswith("\n")
                    and not text.endswith("\t")
                ):
                    text += " "
                text += token

        text = text.strip()
        for special in self.all_special_tokens:
            text = text.replace(" " + special, "")
            text = text.replace(special + " ", "")

        return text.strip()

    def _get_word_ids(
        self, input_ids: List[int], offset_mapping: Optional[List] = None
    ) -> List[int]:
        """Get word IDs for token classification."""
        if offset_mapping is not None:
            word_ids = []
            current_word_idx = 0
            last_end = 0

            for i, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0:
                    word_ids.append(None)
                else:
                    if i > 0 and start > last_end:
                        current_word_idx += 1
                    word_ids.append(current_word_idx)
                    last_end = end

            return word_ids

        tokens = self.convert_ids_to_tokens(input_ids)
        word_ids = []
        current_word_idx = 0

        for i, token in enumerate(tokens):
            if token.startswith("##"):
                word_ids.append(current_word_idx)
            else:
                if i > 0 and not tokens[i - 1].startswith("##"):
                    current_word_idx += 1
                word_ids.append(current_word_idx)

        return word_ids

    def word_ids(self, batch_encoding: Any) -> Optional[List[int]]:
        """Extract word IDs from a batch encoding."""
        if hasattr(batch_encoding, "encodings"):
            word_ids_list = []
            for encoding in batch_encoding.encodings:
                word_ids_list.append(
                    self._get_word_ids(encoding.ids, getattr(encoding, "offsets", None))
                )
            return word_ids_list
        elif isinstance(batch_encoding, dict) and "input_ids" in batch_encoding:
            return self._get_word_ids(
                batch_encoding["input_ids"], batch_encoding.get("offset_mapping")
            )
        return None

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        return_tensors: Optional[str] = None,
        return_token_type_ids: bool = False,
        return_attention_mask: bool = True,
        return_offsets_mapping: bool = False,
    ) -> Dict:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)

        # Add special tokens
        if add_special_tokens:
            if self.bos_token and self.eos_token:
                tokens = [self.bos_token] + tokens + [self.eos_token]
            else:
                tokens = [self.cls_token] + tokens + [self.sep_token]

        input_ids = self.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # Build offset mapping for token classification
        offset_mapping = None
        if return_offsets_mapping:
            offset_mapping = []
            char_pos = 0
            original_text = text

            for token in tokens:
                if token in self.all_special_tokens:
                    offset_mapping.append((0, 0))
                else:
                    token_text = token
                    if token_text.startswith("##"):
                        token_text = token_text[2:]
                    elif token_text in ["Ġ", "Ċ", "ĉ"]:
                        token_text = {"Ġ": " ", "Ċ": "\n", "ĉ": "\t"}.get(token_text, token_text)

                    start = original_text.find(token_text, char_pos)
                    if start == -1:
                        start = char_pos
                    end = start + len(token_text)
                    offset_mapping.append((start, end))
                    char_pos = end

        # Truncation
        if truncation and max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            if offset_mapping:
                offset_mapping = offset_mapping[:max_length]

        # Padding
        if padding and max_length and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            if offset_mapping:
                offset_mapping = offset_mapping + [(0, 0)] * pad_length

        result = {"input_ids": input_ids, "attention_mask": attention_mask}

        if return_offsets_mapping:
            result["offset_mapping"] = offset_mapping

        if return_token_type_ids:
            result["token_type_ids"] = [0] * len(input_ids)

        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch

            result = {
                k: (
                    torch.tensor([v], dtype=torch.long)
                    if k != "offset_mapping"
                    else [torch.tensor(v, dtype=torch.long)]
                )
                for k, v in result.items()
            }
        elif return_tensors == "np":
            import numpy as np

            result = {
                k: np.array([v]) if k != "offset_mapping" else [np.array(v)]
                for k, v in result.items()
            }

        return result

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Decode token IDs back to text."""
        if hasattr(token_ids, "cpu"):
            token_ids = token_ids.cpu().tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]

        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        text = self.convert_tokens_to_string(tokens)

        if clean_up_tokenization_spaces:
            text = " ".join(text.split())

        return text

    def batch_encode_plus(
        self,
        batch_text_or_pairs: List[str],
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
    ) -> Dict:
        """Encode a batch of texts."""
        if isinstance(batch_text_or_pairs, str):
            batch_text_or_pairs = [batch_text_or_pairs]

        batch_output = []
        for text in batch_text_or_pairs:
            encoding = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                padding=False,
                return_tensors=None,
                return_attention_mask=return_attention_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
            )
            batch_output.append(encoding)

        # Apply padding across batch
        if padding:
            max_len = max(len(enc["input_ids"]) for enc in batch_output)
            for enc in batch_output:
                pad_length = max_len - len(enc["input_ids"])
                if pad_length > 0:
                    enc["input_ids"] = enc["input_ids"] + [self.pad_token_id] * pad_length
                    enc["attention_mask"] = enc["attention_mask"] + [0] * pad_length
                    if return_offsets_mapping and "offset_mapping" in enc:
                        enc["offset_mapping"] = enc["offset_mapping"] + [(0, 0)] * pad_length
                    if return_token_type_ids and "token_type_ids" in enc:
                        enc["token_type_ids"] = enc["token_type_ids"] + [0] * pad_length

        batch_result = {
            "input_ids": [enc["input_ids"] for enc in batch_output],
            "attention_mask": [enc["attention_mask"] for enc in batch_output],
        }

        if return_offsets_mapping:
            batch_result["offset_mapping"] = [enc.get("offset_mapping", []) for enc in batch_output]
        if return_token_type_ids:
            batch_result["token_type_ids"] = [enc.get("token_type_ids", []) for enc in batch_output]

        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch

            batch_result = {
                "input_ids": torch.tensor(batch_result["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(batch_result["attention_mask"], dtype=torch.long),
            }
            if return_offsets_mapping:
                batch_result["offset_mapping"] = [
                    torch.tensor(m, dtype=torch.long) for m in batch_result["offset_mapping"]
                ]
            if return_token_type_ids:
                batch_result["token_type_ids"] = torch.tensor(
                    batch_result["token_type_ids"], dtype=torch.long
                )
        elif return_tensors == "np":
            import numpy as np

            batch_result = {
                "input_ids": np.array(batch_result["input_ids"]),
                "attention_mask": np.array(batch_result["attention_mask"]),
            }
            if return_offsets_mapping:
                batch_result["offset_mapping"] = [
                    np.array(m) for m in batch_result["offset_mapping"]
                ]
            if return_token_type_ids:
                batch_result["token_type_ids"] = np.array(batch_result["token_type_ids"])

        return batch_result

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        return_offsets_mapping: bool = True,
        return_token_type_ids: bool = False,
    ) -> Dict:
        """Call method for easy encoding."""
        return self.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_token_type_ids=return_token_type_ids,
        )

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary."""
        return self.vocab

    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        os.makedirs(save_directory, exist_ok=True)

        tokenizer_config = {
            "vocab": self.vocab,
            "special_tokens_map": self.special_tokens_map,
            "tokenizer_type": self.tokenizer_type,
            "merges": self.merges,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "mask_token_id": self.mask_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

        # Save config files
        with open(
            os.path.join(save_directory, "tokenizer_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_directory, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(
            os.path.join(save_directory, "special_tokens_map.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.special_tokens_map, f, ensure_ascii=False, indent=2)

        if self.merges:
            with open(os.path.join(save_directory, "merges.txt"), "w", encoding="utf-8") as f:
                for merge in self.merges:
                    f.write(merge + "\n")

    @classmethod
    def from_pretrained(
        cls, save_directory: str, tokenizer_type: str = "wordpiece", adding_new_token_path: str = "",add_tab_newline_vocab: bool=False
    ):
        """Load tokenizer from directory."""
        with open(os.path.join(save_directory, "vocab.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)

        special_tokens_map = {}
        special_tokens_path = os.path.join(save_directory, "special_tokens_map.json")
        if os.path.exists(special_tokens_path):
            with open(special_tokens_path, "r", encoding="utf-8") as f:
                special_tokens_map = json.load(f)

        # Add new tokens if provided
        if adding_new_token_path and os.path.exists(adding_new_token_path):
            with open(adding_new_token_path, "r", encoding="utf-8") as f:
                adding_new_tokens = f.read().split("\n")

            updated = False
            c = 0
            for ch in adding_new_tokens:
                if ch not in vocab and ch != " ":
                    vocab[ch] = len(vocab)
                    updated = True
                    c += 1

            if updated:
                with open(os.path.join(save_directory, "vocab.json"), "w", encoding="utf-8") as f:
                    json.dump(vocab, f, ensure_ascii=False, indent=2)
                print(f"Updating vocab with {c} new tokens")

        # Load merges for BPE
        merges = []
        merges_path = os.path.join(save_directory, "merges.txt")
        if os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                merges = [line.strip() for line in f.readlines()]

        tokenizer = cls(
            vocab = vocab,
            special_tokens_map = special_tokens_map,
            tokenizer_type = tokenizer_type,
            add_tab_newline_vocab = add_tab_newline_vocab,
            
        )
        tokenizer.merges = merges

        return tokenizer

    def add_tokens(self, new_tokens: Union[str, List[str]]) -> int:
        """Add new tokens to the vocabulary."""
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]

        added_count = 0
        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.inv_vocab[len(self.vocab) - 1] = token
                added_count += 1
        return added_count

    def add_special_tokens(self, special_tokens_dict: Dict[str, str]) -> int:
        """Add special tokens to the tokenizer."""
        added_count = 0
        for key, token in special_tokens_dict.items():
            if hasattr(self, key):
                setattr(self, key, token)
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    self.inv_vocab[len(self.vocab) - 1] = token
                    added_count += 1

        self.all_special_tokens = [
            self.pad_token,
            self.unk_token,
            self.cls_token,
            self.sep_token,
            self.mask_token,
            self.bos_token,
            self.eos_token,
        ]
        self.all_special_ids = [
            self.pad_token_id,
            self.unk_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
            self.bos_token_id,
            self.eos_token_id,
        ]

        return added_count

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
