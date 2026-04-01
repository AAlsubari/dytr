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
Dataset classes for single-task data processing.

This module provides the SingleDatasetProcessing class which handles
dataset preparation for all training strategies.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""



import hashlib
import pickle
from pathlib import Path
#from typing import Optional

#import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dytr.core.config import TrainingStrategy

def process_tags_column(df, tags_column='tags',label_to_ids={},calculate_distribution=False):
    """
    Process tags column: convert string tags to numeric labels.
    
    Args:
        df: DataFrame with tags column
        tags_column: Name of tags column (default: 'tags')
        
    Returns:
        df: DataFrame with processed tags column
        label_to_ids: Dictionary mapping tag labels to IDs
        distribution: Dictionary of tag distribution
    """
    from collections import Counter
    df[tags_column] = df[tags_column].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)
    # Collect all unique tags
    all_tags = set()
    for tags_str in df[tags_column].dropna():
        if isinstance(tags_str, str):
            tags = tags_str.split()
            all_tags.update(tags)
    
    if not label_to_ids:
        # Create label mapping (O -> 0, others start from 1)
        
        label_to_ids = {'O': 0}
        
        for tag in sorted(all_tags):
            if tag != 'O' and not tag.isdigit():
                label_to_ids[tag] = len(label_to_ids)
            elif tag.isdigit() and int(tag) !=0:
                label_to_ids[tag] = int(tag)#len(label_to_ids)
    
    
    # Convert tags to numeric labels
    def convert_tags(tags_str):
        if not isinstance(tags_str, str):
            return ''
        tags = tags_str.split()
        return ' '.join(str(label_to_ids.get(tag, 0)) for tag in tags)
    
    df[tags_column] = df[tags_column].apply(convert_tags)
    if not calculate_distribution:
        return df, label_to_ids, {}
    # Calculate distribution
    distribution = Counter()
    for tags_str in df[tags_column]:
        if tags_str:
            for label in map(int, tags_str.split()):
                distribution[label] += 1
    
    # Convert to readable format
    readable_dist = {label: count for label, count in distribution.items()}
    
    return df, label_to_ids, readable_dist

class SingleDatasetProcessing(Dataset):
    """
    Dataset class for processing single tasks with different training strategies.

    Supports:
    - Sentence Classification
    - Token Classification
    - Seq2Seq
    - Causal Language Modeling

    Args:
        df: DataFrame containing the data
        tokenizer: Tokenizer instance
        max_len: Maximum sequence length
        task_name: Name of the task
        strategy: Training strategy
        num_labels: Number of labels (for classification)
        text_column: Column name for text
        label_column: Column name for labels
        source_column: Column name for source text (seq2seq)
        target_column: Column name for target text (seq2seq)
        tags_column: Column name for token tags
        stride: Stride for windowing (causal LM)
        cache_dir: Directory for caching processed data
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_len,
        task_name,
        strategy,
        num_labels=None,
        text_column="text",
        label_column="label",
        source_column="source",
        target_column="target",
        tags_column="tags",
        stride=None,
        cache_dir="./dataset_cache",
        token_labeling_first_only=True,
        label_to_ids={}
    ):

        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_name = task_name
        self.strategy = strategy
        self.num_labels = num_labels
        self.text_column = text_column
        self.stride = stride if stride is not None else max_len // 2
        self.samples = []
        self.pad_id = tokenizer.pad_token_id or 0
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token_labeling_first_only = token_labeling_first_only
        self.label_to_ids=label_to_ids
        # Process based on strategy
        if strategy == TrainingStrategy.SENTENCE_CLASSIFICATION:
            self._process_classification(label_column)
        elif strategy == TrainingStrategy.TOKEN_CLASSIFICATION:
            self._process_token_classification(tags_column)
        elif strategy == TrainingStrategy.SEQ2SEQ:
            self._process_seq2seq(source_column, target_column)
        elif strategy == TrainingStrategy.CAUSAL_LM:
            self._process_causal_lm(text_column)

        print(f"  Created {len(self.samples)} samples for {self.task_name} >> {strategy.value}")

    def _process_classification(self, label_column):
        """Process data for sentence classification."""
        if self.num_labels is None:
            self.num_labels = len(self.df[label_column].unique())

        for _, row in self.df.iterrows():
            label = int(row[label_column])
            self.samples.append({"text": row[self.text_column], "label": label})
            if len(self.samples) == 1:
                print(f"    Sample text: {str(row[self.text_column])}...")
                print(f"    Sample label: {label}")

    def _process_token_classification(self, tags_column):
        """Process data for token classification."""
        if not self.label_to_ids:
            self.df, self.label_to_ids, _=process_tags_column(self.df, tags_column)
        else:
            self.df, _, _=process_tags_column(self.df, tags_column,label_to_ids=self.label_to_ids)
        """
        all_tags = set()
        for tags in self.df[tags_column]:
            if isinstance(tags, str):
                for tag in tags.split():
                    all_tags.add(int(tag))
            elif isinstance(tags, list):
                for tag in tags:
                    all_tags.add(int(tag))
        all_tags = sorted(all_tags)"""
        self.num_labels = max(self.label_to_ids.values())+1 #len(all_tags)

        valid_samples = 0
        for _, row in self.df.iterrows():
            text = row[self.text_column]
            if isinstance(row[tags_column], str):
                tags = [int(i) for i in row[tags_column].split()]
            else:
                tags = [int(i) for i in row[tags_column]]

            words = text.split()
            if len(tags) == len(words):
                self.samples.append({"text": text, "tags": tags})
                valid_samples += 1
                if valid_samples == 1:
                    print(f"    Sample text: {text}...")
                    print(f"    Sample tags: {tags}...")

    def _process_seq2seq(self, source_column, target_column):
        """Process data for seq2seq."""
        self.num_labels=len(self.tokenizer)
        for _, row in self.df.iterrows():
            self.samples.append({"source": row[source_column], "target": row[target_column]})
            if len(self.samples) == 1:
                print(f"    Sample source: {str(row[source_column])}...")
                print(f"    Sample target: {str(row[target_column])}...")

    def _process_causal_lm(self, text_column):
        """Process data for causal language modeling with windowing."""
        self.num_labels=len(self.tokenizer)
        cache_key = hashlib.md5(
            f"{len(self.df)}_{self.max_len}_{self.stride}_{text_column}_{self.tokenizer.name_or_path}".encode()
        ).hexdigest()
        cache_file = self.cache_dir / f"causal_windows_{cache_key}.pkl"

        if cache_file.exists():
            print(f"  Loading cached window texts from {cache_file}")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
                self.samples = cache_data["window_texts"]
                print(f"    Loaded {len(self.samples)} cached windows")
        else:
            print("  Creating causal LM dataset with dynamic window sampling...")

            if isinstance(self.df, pd.DataFrame):
                texts = self.df[text_column].astype(str).tolist()
            else:
                texts = [t[text_column] for t in self.df]

            for doc_idx, text in enumerate(tqdm(texts, desc="Processing documents")):
                tokens = self.tokenizer.tokenize(text)
                token_count = len(tokens)

                if token_count <= self.max_len:
                    self.samples.append(text)
                else:
                    num_windows = (token_count - self.max_len) // self.stride + 1
                    for window_idx in range(num_windows):
                        start_token = window_idx * self.stride
                        end_token = min(start_token + self.max_len, token_count)

                        window_tokens = tokens[start_token:end_token]
                        if len(window_tokens) < 2:
                            continue
                        window_text = self.tokenizer.convert_tokens_to_string(window_tokens)
                        self.samples.append(window_text)

            # Cache the processed windows
            cache_data = {"window_texts": self.samples}
            response=input("Do you want to save Causal Data Window in cach dir?[Y/n]")
            if response=="Y":
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                print(f"data saved on dir: {cache_file}")

            print(f"    Created and cached {len(self.samples)} windows from {len(texts)} documents")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample based on the strategy."""
        if self.strategy == TrainingStrategy.SENTENCE_CLASSIFICATION:
            return self._get_classification_item(idx)
        elif self.strategy == TrainingStrategy.TOKEN_CLASSIFICATION:
            return self._get_token_classification_item(idx)
        elif self.strategy == TrainingStrategy.SEQ2SEQ:
            return self._get_seq2seq_item(idx)
        elif self.strategy == TrainingStrategy.CAUSAL_LM:
            return self._get_causal_lm_item(idx)

    def _get_classification_item(self, idx):
        """Get item for sentence classification."""
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"]
        if not input_ids:
            input_ids = [self.tokenizer.pad_token_id or 0]
        attention_mask = [1] * len(input_ids)
        labels = torch.full((len(input_ids),), -100, dtype=torch.long)
        labels[0] = sample["label"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": labels,
            "task_name": self.task_name,
            "strategy": self.strategy.value,
        }

    def _get_token_classification_item(self, idx):
        """Get item for token classification."""
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"]
        if not input_ids:
            input_ids = [self.tokenizer.pad_token_id or 0]
        attention_mask = [1] * len(input_ids)

        # Get word ids for token alignment
        try:
            word_ids = self.tokenizer.word_ids(encoding)
        except:
            word_ids = encoding.word_ids()

        labels = [-100] * len(input_ids)
        if self.token_labeling_first_only:
            previous_word_idx = None
            for i, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != previous_word_idx:
                    if word_idx < len(sample["tags"]):
                        tag = sample["tags"][word_idx]
                        label_id = int(tag)
                        if label_id < self.num_labels:
                            labels[i] = label_id
                    previous_word_idx = word_idx
        else:
            
            # Label every token
            for i, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx < len(sample["tags"]):
                    tag = sample["tags"][word_idx]
                    label_id = int(tag)
                    if label_id < self.num_labels:
                        labels[i] = label_id

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "task_name": self.task_name,
            "strategy": self.strategy.value,
        }

    def _get_seq2seq_item(self, idx):
        """Get item for seq2seq."""
        sample = self.samples[idx]
        source_encoding = self.tokenizer(
            sample["source"],
            max_length=self.max_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        target_encoding = self.tokenizer(
            sample["target"],
            max_length=self.max_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )

        return {
            "input_ids": torch.tensor(source_encoding["input_ids"], dtype=torch.long),
            "labels": torch.tensor(target_encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.ones(len(source_encoding["input_ids"]), dtype=torch.long),
            "task_name": self.task_name,
            "strategy": self.strategy.value,
        }

    def _get_causal_lm_item(self, idx):
        """Get item for causal language modeling."""
        window_text = self.samples[idx]
        encoding = self.tokenizer(
            window_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        input_ids = encoding["input_ids"]

        # Shift labels for next token prediction
        labels = input_ids[1:] + [self.pad_id]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "task_name": self.task_name,
            "strategy": self.strategy.value,
        }
