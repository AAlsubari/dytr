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



import logging
import sys
from typing import Optional

_DEFAULT_LOG_LEVEL = logging.INFO
_ROOT_LOGGER_NAME = "dytr"

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger for a module with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        if level is None:
            level = _DEFAULT_LOG_LEVEL
        logger.setLevel(level)
        logger.propagate = False
    
    return logger

def set_log_level(level: int):
    """
    Set log level for all dytr loggers.
    
    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
    """
    logging.getLogger(_ROOT_LOGGER_NAME).setLevel(level)

def disable_logging():
    """Disable all dytr logging."""
    logging.getLogger(_ROOT_LOGGER_NAME).setLevel(logging.WARNING)