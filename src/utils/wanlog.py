# src/utils/wanlog.py

"""
Weights & Biases integration
"""

import wandb
import torch
import numpy as np
from typing import Dict, Any, List, Optional

class WandbLogger:
    """Wandb logger for experiment tracking"""
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        """
        Initialize wandb logger
        
        Args:
            config: Wandb configuration
            model_name: Name of the model being trained
        """
        self.enabled = config.get("enabled", True)
        self.log_frequency = config.get("log_frequency", 10)
        self.log_images = config.get("log_images", True)
        self.log_predictions = config.get("log_predictions", True)
        self.max_images_to_log = config.get("max_images_to_log", 10)
        
        if self.enabled:
            # Initialize wandb run
            wandb.init(
                entity=config.get("entity"),
                project=config.get("project"),
                name=f"{model_name}_{wandb.util.generate_id()}",
                tags=config.get("tags", []) + [model_name],
                notes=config.get("notes", ""),
                config={
                    "model_name": model_name
                }
            )
            
    def update_config(self, config: Dict[str, Any]):
        """Update wandb config with model and training parameters"""
        if self.enabled:
            wandb.config.update(config, allow_val_change=True)
            
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics"""
        if self.enabled:
            wandb.log(metrics, step=step)
            
    def log_batch(self, batch_idx: int, loss: float, step: int):
        """Log batch metrics"""
        if self.enabled and batch_idx % self.log_frequency == 0:
            wandb.log({"batch_loss": loss}, step=step)
            
    def log_epoch(self, epoch: int, train_loss: float, 
                  val_loss: Optional[float] = None,
                  bleu_scores: Optional[Dict[str, float]] = None):
        """Log epoch metrics"""
        if self.enabled:
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss
            }
            
            if val_loss is not None:
                metrics["val_loss"] = val_loss
                
            if bleu_scores is not None:
                for key, value in bleu_scores.items():
                    metrics[key] = value
                    
            wandb.log(metrics)
            
    def log_model_predictions(self, images: torch.Tensor, 
                             true_captions: List[str],
                             pred_captions: List[str],
                             epoch: int):
        """Log model predictions as a table"""
        if self.enabled and self.log_predictions:
            # Create table
            table = wandb.Table(columns=["Image", "True Caption", "Predicted Caption"])
            
            # Add rows (limit to max_images_to_log)
            num_images = min(len(images), self.max_images_to_log)
            for i in range(num_images):
                # Convert tensor to numpy and denormalize
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = self._denormalize_image(img)
                
                # Add to table
                table.add_data(
                    wandb.Image(img),
                    true_captions[i],
                    pred_captions[i]
                )
                
            # Log table
            wandb.log({"test": table})
            
    def log_attention_maps(self, image: np.ndarray, caption: List[str], 
                          attention_weights: List[np.ndarray], epoch: int):
        """Log attention visualization"""
        if self.enabled and self.log_images:
            # Create attention visualizations
            attention_images = []
            
            for i, (word, att_weight) in enumerate(zip(caption, attention_weights)):
                # Create visualization (simplified version)
                fig = self._create_attention_viz(image, word, att_weight)
                attention_images.append(wandb.Image(fig, caption=word))
                
            # Log images
            wandb.log({f"attention_epoch_{epoch}": attention_images})
            
    def _denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """Denormalize image using ImageNet stats"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        return np.clip(image, 0, 1)
    
    def _create_attention_viz(self, image: np.ndarray, word: str, 
                             attention: np.ndarray) -> np.ndarray:
        """Create simple attention visualization"""
        import matplotlib.pyplot as plt
        
        # Resize attention to match image size
        att_size = int(np.sqrt(attention.shape[0]))
        attention = attention.reshape(att_size, att_size)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.imshow(attention, alpha=0.6, cmap='hot', 
                 extent=[0, image.shape[1], image.shape[0], 0])
        ax.set_title(word)
        ax.axis('off')
        
        # Convert to numpy array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img_array
    
    def finish(self):
        """Finish wandb run"""
        if self.enabled:
            wandb.finish()
            
    def log_model_summary(self, model: torch.nn.Module):
        """Log model architecture summary"""
        if self.enabled:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            wandb.run.summary["total_parameters"] = total_params
            wandb.run.summary["trainable_parameters"] = trainable_params