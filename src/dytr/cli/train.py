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
Training CLI for Dynamic Transformers.

This module provides a command-line interface for training models.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Dynamic Transformer model")

    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file (JSON)"
    )

    parser.add_argument("--train_data", type=str, required=True, help="Path to training data file")

    parser.add_argument("--val_data", type=str, help="Path to validation data file (optional)")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments",
        help="Directory to save model checkpoints",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use for training",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("Dynamic Transformers - Training CLI")
    print("=" * 60)

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    with open(args.config, "r") as f:
        config_dict = json.load(f)

    # Load data
    print(f"\nLoading training data from: {args.train_data}")
    import pandas as pd

    train_df = (
        pd.read_csv(args.train_data)
        if args.train_data.endswith(".csv")
        else pd.read_json(args.train_data)
    )

    val_df = None
    if args.val_data:
        print(f"Loading validation data from: {args.val_data}")
        val_df = (
            pd.read_csv(args.val_data)
            if args.val_data.endswith(".csv")
            else pd.read_json(args.val_data)
        )

    # Import library components
    from dytr import (
        DynamicTransformer,
        ModelConfig,
        SingleDatasetProcessing,
        TaskConfig,
        Trainer,
        TrainingStrategy,
        set_seed,
    )

    # Set seed
    set_seed(args.seed)

    # Create configuration
    config = ModelConfig(**config_dict.get("model", {}))
    if args.device:
        config.device = args.device

    # Create task configuration
    task_config = TaskConfig(
        task_name=config_dict.get("task_name", "task"),
        training_strategy=TrainingStrategy(config_dict.get("strategy", "sentence_classification")),
        datasets=[{"train": train_df}],
        num_labels=config_dict.get("num_labels"),
        text_column=config_dict.get("text_column", "text"),
        label_column=config_dict.get("label_column", "label"),
        max_length=config_dict.get("max_length", 256),
    )

    # Initialize model
    print("\nInitializing model...")
    model = DynamicTransformer(config)

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = SingleDatasetProcessing(
        df=train_df,
        tokenizer=model.tokenizer,
        max_len=task_config.max_length,
        task_name=task_config.task_name,
        strategy=task_config.training_strategy,
        num_labels=task_config.num_labels,
        text_column=task_config.text_column,
        label_column=task_config.label_column,
    )

    train_datasets = {task_config.task_name: (train_dataset, task_config.training_strategy)}

    val_datasets = {}
    if val_df is not None:
        val_dataset = SingleDatasetProcessing(
            df=val_df,
            tokenizer=model.tokenizer,
            max_len=task_config.max_length,
            task_name=task_config.task_name,
            strategy=task_config.training_strategy,
            num_labels=task_config.num_labels,
            text_column=task_config.text_column,
            label_column=task_config.label_column,
        )
        val_datasets = {task_config.task_name: (val_dataset, task_config.training_strategy)}

    # Train
    print("Starting training...")
    trainer = Trainer(model, config, exp_dir=args.output_dir)
    model = trainer.train([task_config], train_datasets, val_datasets)

    # Save final model
    output_path = Path(args.output_dir) / "final_model.pt"
    model.save_model(str(output_path))
    print(f"\nModel saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
