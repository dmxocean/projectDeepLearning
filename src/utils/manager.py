# src/utils/manager.py

"""
Configuration manager for handling all project configurations
"""

import os
from typing import Dict, Any
from .config import load_config, get_paths
from .constants import MODELS_DIR, RAW_DATA_DIR, ROOT_DIR

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, debug_mode: bool = None):
        """
        Initialize configuration manager
        
        Args:
            debug_mode: Override debug mode from config
        """
        # Load all configurations
        self.data_config = load_config("data")
        self.model_config = load_config("model")
        self.training_config = load_config("training")
        self.wandb_config = load_config("wandb")
        
        # Set debug mode
        if debug_mode is not None:
            self.debug = debug_mode
        else:
            self.debug = self.data_config["debug"]["enabled"]
            
        # Update paths based on debug mode
        self.dataset_name = self.data_config["dataset"]["name"]
        self.paths = get_paths(self.dataset_name, self.debug)
        
        # Resolve dataset paths relative to project root
        dataset_root = os.path.join(RAW_DATA_DIR, self.data_config["dataset"]["root_dir"])
        self.data_config["dataset"]["root_dir"] = dataset_root
        self.data_config["dataset"]["images_dir"] = os.path.join(
            dataset_root, self.data_config["dataset"]["images_subdir"]
        )
        self.data_config["dataset"]["captions_file"] = os.path.join(
            dataset_root, self.data_config["dataset"]["captions_filename"]
        )
        
        # Update training parameters for debug mode
        if self.debug:
            debug_params = self.training_config["training"]["debug"]
            for key, value in debug_params.items():
                self.training_config["training"][key] = value
                
    def get_model_dir(self, model_type: str) -> str:
        """Get model directory for checkpoints with architecture-specific paths"""
        # Get model config to create architecture-specific path
        model_config = self.get_model_config(model_type)
        encoder_config = self.get_encoder_config()
        
        # Create architecture signature from key parameters
        if model_type == "attention":
            arch_params = [
                f"embed{model_config['embed_size']}",
                f"hidden{model_config['hidden_size']}",
                f"att{model_config['attention_dim']}",
                f"layers{model_config['num_layers']}",
                f"trainCNN{encoder_config['train_cnn']}"
            ]
        elif model_type == "baseline":
            arch_params = [
                f"embed{model_config['embed_size']}",
                f"hidden{model_config['hidden_size']}",
                f"layers{model_config['num_layers']}",
                f"trainCNN{encoder_config['train_cnn']}"
            ]
        else:
            arch_params = ["default"]
            
        arch_signature = "_".join(arch_params)
        
        # Structure results/models/{dataset}/{model_type}/checkpoints/{arch_signature}/
        base_model_dir = os.path.join(MODELS_DIR, self.dataset_name, model_type)
        checkpoint_dir = os.path.join(base_model_dir, "checkpoints", arch_signature)
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        if model_type == "baseline":
            return self.model_config["baseline"]
        elif model_type == "attention":
            return self.model_config["attention"]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def get_encoder_config(self) -> Dict[str, Any]:
        """Get encoder configuration"""
        return self.model_config["encoder"]
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training parameters"""
        return self.training_config["training"]
    
    def get_data_params(self) -> Dict[str, Any]:
        """Get data parameters"""
        return self.data_config
    
    def get_wandb_params(self) -> Dict[str, Any]:
        """Get wandb parameters"""
        return self.wandb_config["wandb"]