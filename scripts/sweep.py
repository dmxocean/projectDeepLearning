# scripts/sweep.py

"""
W&B Sweep runner for hyperparameter optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from typing import Dict, Any

from src.utils.manager import ConfigManager
from src.utils.logger import VerboseLogger
from src.utils.io import load_pickle
from src.models.attention import AttentionCaptionModel
from src.models.baseline import BaselineCaptionModel
from src.preprocessing.dataset import FlickrDataset, create_data_loaders
from src.preprocessing.transforms import get_transforms
from src.preprocessing.vocabulary import Vocabulary
from src.training.trainer import Trainer

logger = VerboseLogger()

def train_sweep(config: Dict[str, Any] = None, model_type: str = "attention"):
    """
    Train function for W&B sweep
    
    Args:
        config: Sweep configuration from W&B
        model_type: Type of model to train ("attention" or "baseline")
    """
    # Initialize wandb run within sweep
    wandb.init()
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Get configurations
    data_config = config_manager.get_data_params()
    model_config = config_manager.get_model_config(model_type)
    encoder_config = config_manager.get_encoder_config()
    training_config = config_manager.get_training_params()
    
    # Override with sweep parameters
    if wandb.config:
        # Update configs with sweep parameters
        if hasattr(wandb.config, 'learning_rate'):
            training_config['learning_rate'] = wandb.config.learning_rate
        if hasattr(wandb.config, 'batch_size'):
            data_config['batch_size'] = wandb.config.batch_size
        if hasattr(wandb.config, 'hidden_size'):
            model_config['hidden_size'] = wandb.config.hidden_size
    
    # Update wandb config with all parameters
    wandb.config.update({
        "model_type": model_type,
        "data_config": data_config,
        "model_config": model_config,
        "encoder_config": encoder_config,
        "training_config": training_config
    })
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load vocabulary
        vocab_path = config_manager.paths["vocab"]
        print(f"Loading vocabulary from {vocab_path}")
        vocab = Vocabulary.load(vocab_path)
        print(f"Vocabulary size: {len(vocab)}")
        
        # Load data splits
        splits_path = config_manager.paths["splits"]
        print(f"Loading data splits from {splits_path}")
        splits = load_pickle(splits_path)
        train_df = splits['train']
        val_df = splits['val']
        
        # Get transforms
        train_transform, val_transform = get_transforms(
            resize=data_config["image"]["resize_size"],
            crop=data_config["image"]["crop_size"]
        )
        
        # Create datasets
        print("Creating datasets...")
        images_dir = data_config["dataset"]["images_dir"]
        
        train_dataset = FlickrDataset(
            data_df=train_df,
            root_dir=images_dir,
            vocab=vocab,
            transform=train_transform
        )
        
        val_dataset = FlickrDataset(
            data_df=val_df,
            root_dir=images_dir,
            vocab=vocab,
            transform=val_transform
        )
        
        # Create data loaders manually since sweep doesn't need test loader
        print("Creating data loaders...")
        batch_size = training_config["batch_size"]
        
        from torch.utils.data import DataLoader
        from src.preprocessing.dataset import FlickrCollate
        
        collate_fn = FlickrCollate(pad_idx=vocab.stoi["<PAD>"])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Initialize model
        print(f"Initializing {model_type} model...")
        if model_type == "attention":
            model = AttentionCaptionModel(
                embed_size=model_config["embed_size"],
                hidden_size=model_config["hidden_size"],
                vocab_size=len(vocab),
                attention_dim=model_config["attention_dim"],
                num_layers=model_config["num_layers"],
                dropout=model_config["dropout"],
                **encoder_config
            )
        else:
            model = BaselineCaptionModel(
                embed_size=model_config["embed_size"],
                hidden_size=model_config["hidden_size"],
                vocab_size=len(vocab),
                num_layers=model_config["num_layers"],
                dropout=model_config["dropout"],
                **encoder_config
            )
        
        # Move to device
        model = model.to(device)
        
        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        wandb.run.summary["total_parameters"] = total_params
        
        # Initialize optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # Initialize scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=training_config['lr_scheduler_factor'],
            patience=training_config['lr_scheduler_patience']
        )
        
        # Initialize criterion
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
        
        # Create unique model paths for each hyperparameter combination
        run_id = wandb.run.id if wandb.run else "default"
        # Use config manager to get architecture-specific directory
        checkpoint_dir = config_manager.get_model_dir(model_type)
        model_paths = {
            'checkpoint_path': os.path.join(checkpoint_dir, f'sweep_{run_id}_checkpoint.pth'),
            'best_model_path': os.path.join(checkpoint_dir, f'sweep_{run_id}_best.pth')
        }
        
        # Initialize trainer without WandbLogger (sweep already initialized wandb)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=training_config,
            device=device,
            vocab=vocab,
            model_paths=model_paths,
            wandb_logger=None  # Will use existing wandb run
        )
        
        # Train model
        print("Starting training...")
        trained_model, history = trainer.train()
        
        # Generate wandb table with test predictions
        if wandb.run is not None:
            print("Generating wandb predictions table...")
            trainer._log_sample_predictions(trainer.num_epochs)
        
        # Log final metrics
        if history['val_losses']:
            wandb.run.summary['best_val_loss'] = min(history['val_losses'])
        if history['bleu_scores']:
            best_bleu = max([scores.get('bleu4', 0) for scores in history['bleu_scores']])
            wandb.run.summary['best_bleu4'] = best_bleu
        
    except Exception as e:
        print(f"Error during sweep training: {e}")
        raise
    finally:
        wandb.finish()

def run_sweep(model_type: str = "attention", count: int = 10):
    """
    Initialize and run W&B sweep
    
    Args:
        model_type: Type of model to train ("attention" or "baseline")
        count: Number of sweep runs to execute
    """
    # Load configurations
    config_manager = ConfigManager()
    wandb_config = config_manager.get_wandb_params()  # Access the full wandb config including sweep
    
    # Check if sweep is enabled
    if not wandb_config.get('sweep', {}).get('enabled', False):
        print("Sweep is not enabled in wandb.yaml. Set sweep.enabled to true.")
        return
    
    # Extract sweep configuration
    sweep_config = {
        "method": wandb_config['sweep']['method'],
        "metric": wandb_config['sweep']['metric'],
        "parameters": wandb_config['sweep']['parameters']
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        entity=wandb_config.get('entity'),
        project=wandb_config.get('project')
    )
    
    print(f"Created sweep with ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/{wandb_config.get('entity')}/{wandb_config.get('project')}/sweeps/{sweep_id}")
    
    # Run sweep agent
    train_fn = partial(train_sweep, model_type=model_type)
    wandb.agent(sweep_id, train_fn, count=count)
    
def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run W&B sweep for hyperparameter optimization")
    parser.add_argument(
        "--model",
        type=str,
        choices=["attention", "baseline"],
        default="attention",
        help="Model type to train"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of sweep runs to execute"
    )
    
    args = parser.parse_args()
    
    # Run sweep
    run_sweep(model_type=args.model, count=args.count)

if __name__ == "__main__":
    main()