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
Export CLI for Dynamic Transformers.

This module provides a command-line interface for exporting single-task models.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a single-task model from a multi-task model"
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the saved multi-task model"
    )

    parser.add_argument("--task_name", type=str, required=True, help="Name of the task to export")

    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the exported model"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use for loading the model",
    )

    return parser.parse_args()


def main():
    """Main export function."""
    args = parse_args()

    print("=" * 60)
    print("Dynamic Transformers - Export CLI")
    print("=" * 60)

    print(f"\nLoading model from: {args.model_path}")
    print(f"Exporting task: {args.task_name}")
    print(f"Saving to: {args.output_path}")

    # Import library components
    from dytr import DynamicTransformer

    # Load model
    model = DynamicTransformer.load_model(args.model_path, device=args.device)

    if model is None:
        print("Error: Failed to load model")
        sys.exit(1)

    # Export task
    exporter = model.get_exporter()
    exported_model = exporter.export_single_task(args.task_name, args.output_path)

    print(f"\nSuccessfully exported {args.task_name} to {args.output_path}")

    print("\n" + "=" * 60)
    print("Export completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
