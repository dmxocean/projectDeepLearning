# src/training/metrics.py

"""
Evaluation metrics for image captioning (BLEU, CIDEr)
"""

import torch
from typing import List, Dict
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm

from ..preprocessing.vocabulary import Vocabulary
from ..utils.constants import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN

def calculate_bleu(model, data_loader, vocab: Vocabulary, device: torch.device, 
                  max_samples: int = None) -> Dict[str, float]:
    """
    Calculate BLEU score on a dataset
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for the dataset
        vocab: Vocabulary object
        device: Device to run on
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        bleu_scores: Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    # Set model to evaluation mode
    model.eval()
    
    # Determine if this is an attention model
    is_attention_model = hasattr(model, 'caption_image_with_attention')
    
    # Initialize lists for references and hypotheses
    references = []
    hypotheses = []
    
    # No gradient calculation needed
    with torch.no_grad():
        # Progress bar
        total = min(len(data_loader), max_samples // data_loader.batch_size + 1) if max_samples else len(data_loader)
        with tqdm(total=total, desc="Calculating BLEU") as pbar:
            # Iterate over batches
            for i, (images, captions, lengths) in enumerate(data_loader):
                # Check if we've processed enough samples
                if max_samples and i * data_loader.batch_size >= max_samples:
                    break
                
                # Move to device
                images = images.to(device)
                
                # Get predictions based on model type
                if is_attention_model:
                    predictions, _ = model.sample(images)
                else:
                    predictions = model.sample(images)
                
                # Process each image in the batch
                for j in range(images.size(0)):
                    # Check if we've processed enough samples
                    if max_samples and len(hypotheses) >= max_samples:
                        break
                    
                    # Get predicted caption
                    pred_tokens = []
                    for token_idx in predictions[j]:
                        token = vocab.itos[token_idx.item()]
                        if token == EOS_TOKEN:
                            break
                        if token not in [PAD_TOKEN, SOS_TOKEN]:
                            pred_tokens.append(token)
                    
                    # Get reference caption
                    ref_tokens = []
                    for token_idx in captions[j]:
                        token = vocab.itos[token_idx.item()]
                        if token == EOS_TOKEN:
                            break
                        if token not in [PAD_TOKEN, SOS_TOKEN]:
                            ref_tokens.append(token)
                    
                    # Add to lists
                    references.append([ref_tokens])  # Each reference is a list of reference translations
                    hypotheses.append(pred_tokens)
                
                # Update progress bar
                pbar.update(1)
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    return {
        "bleu1": bleu1 * 100,
        "bleu2": bleu2 * 100,
        "bleu3": bleu3 * 100,
        "bleu4": bleu4 * 100
    }

def calculate_cider(references: List[List[str]], hypotheses: List[str]) -> float:
    """
    Calculate CIDEr score (simplified version without full implementation)
    
    Args:
        references: List of reference captions (each item is a list of tokens)
        hypotheses: List of hypothesis captions (each item is a list of tokens)
        
    Returns:
        cider_score: CIDEr score
    """
    # This is a placeholder for CIDEr calculation
    # Full CIDEr implementation requires additional dependencies
    # For now, return average of BLEU scores as approximation
    try:
        from pycocoevalcap.cider.cider import Cider
        
        # Convert to required format
        refs = {i: [' '.join(ref) for ref in references[i]] for i in range(len(references))}
        hyps = {i: [' '.join(hypotheses[i])] for i in range(len(hypotheses))}
        
        # Calculate CIDEr
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(refs, hyps)
        return score * 100
    except ImportError:
        # Return approximation using BLEU-4 if CIDEr not available
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        return bleu4 * 100

def calculate_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                  criterion: torch.nn.Module, pad_idx: int) -> tuple:
    """
    Calculate loss with attention to padding
    
    Args:
        predictions: Model predictions (batch_size * seq_length, vocab_size)
        targets: Target indices (batch_size * seq_length)
        criterion: Loss criterion
        pad_idx: Padding index
        
    Returns:
        loss: Calculated loss
        n_tokens: Number of non-padding tokens
    """
    # Create a mask to exclude padding tokens from loss
    non_pad_mask = (targets != pad_idx)
    
    # Count non-padding tokens
    n_tokens = non_pad_mask.sum().item()
    
    # Calculate loss
    loss = criterion(predictions, targets)
    
    return loss, n_tokens