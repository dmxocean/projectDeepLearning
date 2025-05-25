# src/utils/logger.py

"""
Verbose logging utility
"""

import time
import torch
from typing import Optional

class VerboseLogger:
    """Logger for training progress with detailed output"""
    
    def __init__(self, print_frequency: int = 50):
        """
        Initialize logger
        
        Args:
            print_frequency: How often to print progress
        """
        self.print_frequency = print_frequency
        self.start_time = None
        self.epoch_start_time = None
        
    def start_training(self, num_epochs: int, model_name: str):
        """Log training start"""
        self.start_time = time.time()
        print(f"\nTraining {model_name} model for {num_epochs} epochs...")
        print()
        
    def start_epoch(self, epoch: int, num_epochs: int):
        """Log epoch start"""
        self.epoch_start_time = time.time()
        print(f"Epoch {epoch}/{num_epochs}")
        print()
        
    def log_batch(self, batch_idx: int, num_batches: int, loss: float, 
                  batch_time: float):
        """Log batch progress"""
        if (batch_idx + 1) % self.print_frequency == 0:
            elapsed = time.time() - self.epoch_start_time
            eta = (num_batches - batch_idx - 1) * batch_time
            print(f"Batch {batch_idx+1}/{num_batches} | "
                  f"Loss: {loss:.4f} | "
                  f"Time: {batch_time:.3f}s/batch | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"ETA: {eta:.1f}s")
    
    def log_epoch_end(self, epoch: int, train_loss: float, 
                      val_loss: Optional[float] = None):
        """Log epoch completion"""
        epoch_time = time.time() - self.epoch_start_time
        msg = f"Epoch {epoch} - Train loss: {train_loss:.4f}"
        if val_loss is not None:
            msg += f", Val loss: {val_loss:.4f}"
        msg += f" | Time: {epoch_time:.1f}s"
        print(msg)
        
    def log_bleu_scores(self, bleu_scores: dict):
        """Log BLEU scores"""
        print(f"BLEU scores - "
              f"BLEU-1: {bleu_scores['bleu1']:.2f}, "
              f"BLEU-2: {bleu_scores['bleu2']:.2f}, "
              f"BLEU-3: {bleu_scores['bleu3']:.2f}, "
              f"BLEU-4: {bleu_scores['bleu4']:.2f}")
        
    def log_training_end(self):
        """Log training completion"""
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"Training completed in {total_time/60:.1f} minutes")
            print()
            
    def debug_tensor(self, name: str, tensor: torch.Tensor, level: int = 0):
        """Log tensor debug information"""
        indent = "  " * level
        print(f"{indent}DEBUG: {name}")
        print(f"{indent}  Shape: {tensor.shape}")
        print(f"{indent}  Type: {tensor.dtype}")
        print(f"{indent}  Device: {tensor.device}")
        print(f"{indent}  Values - Min: {tensor.min().item():.4f}, "
              f"Max: {tensor.max().item():.4f}, "
              f"Mean: {tensor.mean().item():.4f}")
        print(f"{indent}  Requires grad: {tensor.requires_grad}")

class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count