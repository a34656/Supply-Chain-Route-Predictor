"""
Universal Import Resolver for AI Engine
Works no matter where you run the code from.
"""

import sys
from pathlib import Path


def setup_imports():
    """
    Ensure project root and src/ are always available for imports.
    """

    # Path to this file: ai_engine/src/utils/import_helper.py
    current_file = Path(__file__).resolve()

    # ai_engine/src/utils
    utils_dir = current_file.parent

    # ai_engine/src
    src_dir = utils_dir.parent

    # ai_engine/
    project_root = src_dir.parent

    paths_to_add = [
        str(project_root),
        str(src_dir),
    ]

    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)


def safe_import(module_path: str):
    """
    Dynamically import modules from anywhere.

    Example:
        safe_import("src.data_preprocessing.clean_wgi")
    """
    setup_imports()

    try:
        module = __import__(module_path, fromlist=['*'])
        return module

    except ImportError as e:
        raise ImportError(
            f"Failed to import {module_path}. "
            f"sys.path = {sys.path}"
        ) from e


# Run on import
setup_imports()