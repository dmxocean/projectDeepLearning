# src/comparison/evaluator.py

"""
Model comparison and evaluation utilities
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Standardized visualization settings
STANDARD_DPI = 300
STANDARD_FIGSIZE = (10, 8)
FONT_TITLE = 16
FONT_LABEL = 14
FONT_TEXT = 12

from ..training.metrics import calculate_bleu, calculate_cider
from ..preprocessing.vocabulary import Vocabulary
from ..utils.io import save_json, load_json
from ..utils.constants import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN

class ModelEvaluator:
    """Evaluate and compare multiple models"""
    
    def __init__(self, models: Dict[str, torch.nn.Module], vocab: Vocabulary, 
                 device: torch.device):
        """
        Initialize evaluator
        
        Args:
            models: Dictionary of model_name -> model
            vocab: Vocabulary object
            device: Device to run on
        """
        self.models = models
        self.vocab = vocab
        self.device = device
        
        # Put all models in eval mode
        for model in self.models.values():
            model.eval()
    
    def evaluate_all(self, test_loader, max_samples: int = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on test set
        
        Args:
            test_loader: Test data loader
            max_samples: Maximum samples to evaluate
            
        Returns:
            results: Dictionary of model_name -> metrics
        """
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name} model...")
            
            # Calculate BLEU scores
            bleu_scores = calculate_bleu(model, test_loader, self.vocab, 
                                       self.device, max_samples)
            
            # Calculate additional metrics if needed
            results[model_name] = {
                **bleu_scores,
                "model_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            print(f"{model_name} - BLEU-1: {bleu_scores['bleu1']:.2f}, "
                  f"BLEU-2: {bleu_scores['bleu2']:.2f}, "
                  f"BLEU-3: {bleu_scores['bleu3']:.2f}, "
                  f"BLEU-4: {bleu_scores['bleu4']:.2f}")
        
        return results
    
    def compare_predictions(self, images: torch.Tensor, captions: torch.Tensor) -> pd.DataFrame:
        """
        Compare predictions from all models
        
        Args:
            images: Batch of images
            captions: Ground truth captions
            
        Returns:
            comparison_df: DataFrame with predictions from all models
        """
        images = images.to(self.device)
        predictions = {}
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                # Generate predictions
                if hasattr(model, 'caption_image_with_attention'):
                    # Attention model
                    batch_predictions = []
                    for i in range(images.size(0)):
                        img = images[i].unsqueeze(0)
                        caption, _ = model.caption_image_with_attention(img, self.vocab)
                        batch_predictions.append(caption)
                    predictions[model_name] = batch_predictions
                else:
                    # Baseline model
                    batch_predictions = []
                    for i in range(images.size(0)):
                        img = images[i].unsqueeze(0)
                        caption = model.caption_image(img, self.vocab)
                        batch_predictions.append(caption)
                    predictions[model_name] = batch_predictions
        
        # Convert ground truth captions
        gt_captions = []
        for i in range(captions.size(0)):
            gt_words = []
            for token_idx in captions[i]:
                token = self.vocab.itos[token_idx.item()]
                if token == EOS_TOKEN:
                    break
                if token not in [PAD_TOKEN, SOS_TOKEN]:
                    gt_words.append(token)
            gt_captions.append(' '.join(gt_words))
        
        # Create comparison dataframe
        data = {"ground_truth": gt_captions}
        data.update(predictions)
        
        return pd.DataFrame(data)
    
    def plot_metrics_comparison(self, results: Dict[str, Dict[str, float]], 
                               save_path: str = None):
        """
        Plot comparison of metrics across models
        
        Args:
            results: Results dictionary from evaluate_all
            save_path: Optional path to save figure
        """
        # Extract metrics for plotting
        models = list(results.keys())
        metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4']
        
        # Create data for plotting
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric.upper(),
                    'Score': results[model][metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create plot with standardized settings
        plt.figure(figsize=STANDARD_FIGSIZE)
        sns.barplot(x='Metric', y='Score', hue='Model', data=df)
        plt.title('Model Performance Comparison', fontsize=FONT_TITLE)
        plt.xlabel('Metric', fontsize=FONT_LABEL)
        plt.ylabel('Score (%)', fontsize=FONT_LABEL)
        plt.legend(title='Model', fontsize=FONT_TEXT, title_fontsize=FONT_TEXT)
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=FONT_TEXT)
        plt.yticks(fontsize=FONT_TEXT)
        
        if save_path:
            plt.savefig(save_path, dpi=STANDARD_DPI, bbox_inches='tight', facecolor='white')
        
        plt.close()
    
    def analyze_caption_lengths(self, test_loader, max_samples: int = 1000) -> pd.DataFrame:
        """
        Analyze caption lengths generated by each model
        
        Args:
            test_loader: Test data loader
            max_samples: Maximum samples to analyze
            
        Returns:
            length_df: DataFrame with caption length statistics
        """
        length_data = {model_name: [] for model_name in self.models}
        gt_lengths = []
        
        sample_count = 0
        
        with torch.no_grad():
            for images, captions, _ in test_loader:
                if max_samples and sample_count >= max_samples:
                    break
                
                images = images.to(self.device)
                
                # Get predictions from each model
                for model_name, model in self.models.items():
                    if hasattr(model, 'sample'):
                        # Get predictions
                        if hasattr(model, 'caption_image_with_attention'):
                            predictions, _ = model.sample(images)
                        else:
                            predictions = model.sample(images)
                        
                        # Count words in each caption
                        for i in range(predictions.size(0)):
                            word_count = 0
                            for token_idx in predictions[i]:
                                token = self.vocab.itos[token_idx.item()]
                                if token == EOS_TOKEN:
                                    break
                                if token not in [PAD_TOKEN, SOS_TOKEN]:
                                    word_count += 1
                            length_data[model_name].append(word_count)
                
                # Count ground truth lengths
                for i in range(captions.size(0)):
                    word_count = 0
                    for token_idx in captions[i]:
                        token = self.vocab.itos[token_idx.item()]
                        if token == EOS_TOKEN:
                            break
                        if token not in [PAD_TOKEN, SOS_TOKEN]:
                            word_count += 1
                    gt_lengths.append(word_count)
                
                sample_count += images.size(0)
        
        # Create statistics
        stats = []
        
        # Ground truth statistics
        stats.append({
            'Model': 'Ground Truth',
            'Mean Length': np.mean(gt_lengths),
            'Std Length': np.std(gt_lengths),
            'Min Length': np.min(gt_lengths),
            'Max Length': np.max(gt_lengths),
            'Median Length': np.median(gt_lengths)
        })
        
        # Model statistics
        for model_name, lengths in length_data.items():
            if lengths:
                stats.append({
                    'Model': model_name,
                    'Mean Length': np.mean(lengths),
                    'Std Length': np.std(lengths),
                    'Min Length': np.min(lengths),
                    'Max Length': np.max(lengths),
                    'Median Length': np.median(lengths)
                })
        
        return pd.DataFrame(stats)
    
    def save_results(self, results: Dict, filepath: str):
        """Save evaluation results to file"""
        save_json(results, filepath)
        print(f"Saved results to {filepath}")
    
    def load_results(self, filepath: str) -> Dict:
        """Load evaluation results from file"""
        return load_json(filepath)