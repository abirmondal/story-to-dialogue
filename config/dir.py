"""
config/dir.py

This module contains configuration settings for directory paths used in the project.
"""

from pathlib import Path

# Define the base directory as the parent directory of this file's parent
BASE_DIR = Path(__file__).resolve().parent.parent

# SODA dataset HuggingFace Repository
SODA_HF_REPO = "allenai/soda"

# SODA Bengali dataset HuggingFace Repository
SODA_BENGALI_HF_REPO = "abirmondalind/soda_bengali_small"

# Directory to save prediction files
PREDICTIONS_DIR = BASE_DIR / "predictions"

def update_directories(base_dir: str) -> None:
    """
    Updates directory paths based on a new base directory.

    Args:
        base_dir (str): The new base directory path.
    """
    global BASE_DIR, PREDICTIONS_DIR
    BASE_DIR = Path(base_dir).resolve()
    PREDICTIONS_DIR = BASE_DIR / "predictions"