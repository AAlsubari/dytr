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
Training utilities for Dynamic Transformers.

This module provides helper functions for training, including seed setting
and dataset processing for QA tasks.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""



import random
from typing import Dict, List, Optional

import numpy as np
#import pandas as pd
import torch


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def process_qa_dataset(
    ds,
    config,
    num_samples: int = 200000,
    text_column: Optional[str] = None,
    conversations_col: Optional[str] = None,
    question_col: Optional[str] = None,
    answer_col: Optional[str] = None,
    min_text_length: int = 50, 
) -> List[Dict[str, str]]:
    """
    Process question-answering datasets into text format for language modeling.

    This function processes QA datasets from various formats (conversations,
    question-answer pairs) into a unified text format suitable for training.

    Args:
        ds: Dataset (list of dictionaries or DataFrame)
        config: Model configuration (for special tokens)
        num_samples: Maximum number of samples to process
        text_column: Column containing text (if applicable)
        conversations_col: Column containing conversation history
        question_col: Column containing questions
        answer_col: Column containing answers
        min_text_length: Minimum text length to include

    Returns:
        List of processed text samples
    """
    processed_texts = []
    all_processed_texts = []
    answer_start = config.special_tokens.get("answer_start", "<|answer|>")

    c = 0
    for i, item in enumerate(ds):
        if c >= 20000:
            c = 0
            all_processed_texts.extend([{"text": t.lower()} for t in processed_texts])
            processed_texts = []

        try:
            # Handle conversation format
            if conversations_col in item and isinstance(item[conversations_col], list):
                conversations = item[conversations_col]
                if len(conversations) >= 2:
                    human_text = ""
                    gpt_text = ""
                    for conv in conversations:
                        if isinstance(conv, dict):
                            from_val = conv.get("from", "").lower()
                            value_val = conv.get("value", "")
                            if "human" in from_val or "user" in from_val or "question" in from_val:
                                human_text = value_val
                            elif (
                                "gpt" in from_val or "assistant" in from_val or "answer" in from_val
                            ):
                                gpt_text = value_val

                    if human_text and gpt_text:
                        merged = f"Q: {human_text} \n {answer_start} \n {gpt_text} \n \n"
                        if len(merged) >= min_text_length:
                            processed_texts.append(merged)
                            c += len(merged)

            # Handle question-answer format
            elif (
                question_col
                and answer_col
                and question_col in item
                and item[question_col]
                and answer_col in item
                and item[answer_col]
            ):
                question = item[question_col]
                if text_column and text_column in item and item[text_column]:
                    question = question + f"\n {item[text_column]}"

                answer = item[answer_col]
                if question and answer and isinstance(question, str) and isinstance(answer, str):
                    merged = f"Q: {question} \n {answer_start} \n {answer} \n \n"
                    if len(merged) >= min_text_length:
                        processed_texts.append(merged)
                        c += len(merged)

            # Try to auto-detect question and answer columns
            else:
                question = None
                answer = None
                question_candidates = [
                    "question",
                    "Question",
                    "query",
                    "Query",
                    "input",
                    "Input",
                    "prompt",
                    "Prompt",
                    "instruction",
                    "Instruction",
                ]
                answer_candidates = [
                    "answer",
                    "Answer",
                    "response",
                    "Response",
                    "output",
                    "Output",
                    "completion",
                    "Completion",
                    "target",
                    "Target",
                ]

                for q_col in question_candidates:
                    if q_col in item and isinstance(item[q_col], str) and item[q_col].strip():
                        question = item[q_col]
                        break

                for a_col in answer_candidates:
                    if a_col in item and isinstance(item[a_col], str) and item[a_col].strip():
                        answer = item[a_col]
                        break

                if not question and not answer and text_column in item and item[text_column]:
                    text = item[text_column]
                    if isinstance(text, str) and len(text.strip()) >= min_text_length:
                        processed_texts.append(text + "\n\n")
                        c += len(text)

                elif question and answer:
                    merged = f"Q: {question} \n {answer_start} \n {answer} \n \n"
                    if len(merged) >= min_text_length:
                        processed_texts.append(merged)
                        c += len(merged)

            if i == 0 and processed_texts:
                print(f"  Sample processed text: {processed_texts[0]}...")

        except Exception as e:
            print(f"  Error in QA processing: {e}")
            continue

    if processed_texts:
        all_processed_texts.extend([{"text": t.lower()} for t in processed_texts])
        #all_processed_texts.append({"text": "\n".join(processed_texts)})
        print(f"Total Questions Answers: {len(all_processed_texts)}")

    return all_processed_texts
