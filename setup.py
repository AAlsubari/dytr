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
Setup script for Dynamic Transformers library.

This file provides backward compatibility with older pip versions
and can be used as an alternative to pyproject.toml.

Author: Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""

from setuptools import setup, find_packages
import os
import sys

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from __init__.py
version = {}
with open(os.path.join(this_directory, 'src', 'dytr', '__init__.py'), encoding='utf-8') as f:
    for line in f:
        if line.startswith('__version__'):
            exec(line, version)
            break

setup(
    name="dytr",
    version=version.get('__version__', '0.1.0'),
    author="Akram Alsubari",
    author_email="akram.alsubari@outlook.com",
    maintainer="Akram Alsubari",
    maintainer_email="akram.alsubari87@gmail.com",
    description="Dynamic Transformer for Multi-Task Learning with Continual Learning Support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",  # No GitHub URL
    project_urls={
        "Bug Reports": "http://www.linkedin.com/in/akram-alsubari",
        "Source": "",
        "Documentation": "",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "transformers": ["transformers>=4.0.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "build>=0.7",
            "twine>=3.4",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "all": [
            "transformers>=4.0.0",
            "pytest>=6.0",
            "sphinx>=4.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "dytr-train=dytr.cli.train:main",
            "dytr-export=dytr.cli.export:main",
        ],
    },
)