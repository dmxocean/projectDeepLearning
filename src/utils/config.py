# src/utils/config.py

"""
Configuration loading utilities
"""

import os
import yaml
from typing import Dict, Any
from .constants import CONFIG_DIR, PROCESSED_DATA_DIR

def load_yaml(filepath: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load configuration file by name
    
    Args:
        config_name: Name of config file (without .yaml extension)
        
    Returns:
        config: Configuration dictionary
    """
    config_path = os.path.join(CONFIG_DIR, f"{config_name}.yaml")
    return load_yaml(config_path)

def get_paths(dataset_name: str, debug: bool = False) -> Dict[str, str]:
    """
    Get dataset paths based on debug mode
    
    Args:
        dataset_name: Name of the dataset
        debug: Whether in debug mode
        
    Returns:
        paths: Dictionary of paths
    """
    if debug:
        base_dir = os.path.join(PROCESSED_DATA_DIR, "debug", dataset_name)
    else:
        base_dir = os.path.join(PROCESSED_DATA_DIR, dataset_name)
    
    return {
        "processed": base_dir,
        "vocab": os.path.join(base_dir, "vocab.pkl"),
        "splits": os.path.join(base_dir, "splits.pkl")
    }