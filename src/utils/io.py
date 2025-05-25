# src/utils/io.py

"""
Input/output utilities
"""

import os
import pickle
import json
import torch
from typing import Any, Dict

def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)

def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> Any:
    """Load object from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_json(obj: Dict, filepath: str) -> None:
    """Save dictionary to JSON file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=4)

def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_checkpoint(state: Dict, filepath: str, is_best: bool = False) -> None:
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model state, optimizer state, etc
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    ensure_dir(os.path.dirname(filepath))
    torch.save(state, filepath)
    
    if is_best:
        print(f"Saved best model to {filepath}")
    else:
        print(f"Saved checkpoint to {filepath}")

def load_checkpoint(filepath: str, device: torch.device) -> Dict:
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint
        device: Device to load checkpoint to
        
    Returns:
        checkpoint: Dictionary containing model state, optimizer state, etc
    """
    return torch.load(filepath, map_location=device)

def check_model_availability(checkpoint_path: str, best_model_path: str) -> tuple:
    """
    Check if a trained model or checkpoint exists
    
    Args:
        checkpoint_path: Path to checkpoint
        best_model_path: Path to best model
        
    Returns:
        status: "trained", "checkpoint", or "train_new"
        checkpoint: Loaded checkpoint or None
    """
    # Check for fully trained model
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, weights_only=False)
        print(f"Found trained model at {best_model_path}")
        return "trained", checkpoint
    
    # Check for checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"Found checkpoint at {checkpoint_path}")
        return "checkpoint", checkpoint
    
    # No model found
    print("No trained model or checkpoint found. Will train a new model")
    return "train_new", None