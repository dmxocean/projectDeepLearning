# src/visualization/attention.py

"""
Attention map visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Optional
import os

# Standardized settings matching captioning.py
STANDARD_DPI = 300
STANDARD_FIGSIZE = (14, 14)  # Square format
ATTENTION_GRID_SIZE = 3      # Size for each subplot in grid
FONT_TITLE = 16
FONT_TEXT = 12

def visualize_attention(image: np.ndarray, caption: List[str], 
                       attention_weights: List[torch.Tensor], 
                       show_every: int = 1, save_path: Optional[str] = None):
    """
    Visualize attention for each word in the caption
    
    Args:
        image: Original image as numpy array
        caption: List of caption words
        attention_weights: List of attention weight tensors
        show_every: Show every nth word
        save_path: Optional path to save the figure
    """
    # Create figure with subplots - one subplot per word, plus one for the original image
    words_to_show = list(range(0, len(caption), show_every))
    num_words = len(words_to_show)
    
    # Determine subplot grid size
    if num_words < 6:
        # For few words, use 1 row
        nrows = 1
        ncols = num_words + 1  # +1 for original image
    else:
        # For more words, use 2 or more rows
        ncols = min(5, num_words + 1)
        nrows = (num_words + 1 + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*ATTENTION_GRID_SIZE, nrows*ATTENTION_GRID_SIZE))
    
    # Handle different subplot layouts correctly
    if nrows == 1 and ncols == 1:
        # Single subplot
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        # 1D array of subplots (single row or column)
        axes_flat = axes.flatten()
    else:
        # 2D array of subplots
        axes_flat = axes.flatten()
    
    # Plot original image
    axes_flat[0].imshow(image)
    axes_flat[0].set_title('Original Image', fontsize=FONT_TEXT)
    axes_flat[0].axis('off')
    
    # Plot attention for each selected word
    for idx, word_idx in enumerate(words_to_show):
        if idx + 1 >= len(axes_flat):  # Ensure we don't run out of subplots
            break
            
        # Get attention weights and word
        att_weight = attention_weights[word_idx]
        word = caption[word_idx]
        
        # Reshape attention weights to square for visualization
        # Assuming it's a square feature map
        att_size = int(np.sqrt(att_weight.shape[0]))
        att_weight = att_weight.reshape(att_size, att_size)
        
        # Resize attention map to match image size
        h, w = image.shape[:2]
        att_weight = np.repeat(np.repeat(att_weight, h//att_size, axis=0), w//att_size, axis=1)
        att_weight = att_weight[:h, :w]  # Crop to match image size
        
        # Plot the word-specific attention
        axes_flat[idx + 1].imshow(image)
        axes_flat[idx + 1].imshow(att_weight, alpha=0.6, cmap='hot')
        axes_flat[idx + 1].set_title(word, fontsize=FONT_TEXT)
        axes_flat[idx + 1].axis('off')
    
    # Hide any unused subplots
    for i in range(len(words_to_show) + 1, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=STANDARD_DPI, facecolor='white')
    
    plt.close()

def visualize_single_attention(image: np.ndarray, word: str, 
                              attention_weight: torch.Tensor, 
                              save_path: Optional[str] = None,
                              figsize: tuple = STANDARD_FIGSIZE):
    """
    Visualize attention for a single word with better aspect ratio
    
    Args:
        image: Original image as numpy array
        word: The word being attended to
        attention_weight: Attention weight tensor for this word
        save_path: Optional path to save the figure
        figsize: Figure size in inches (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Reshape attention weights
    att_size = int(np.sqrt(attention_weight.shape[0]))
    att_weight = attention_weight.reshape(att_size, att_size)
    
    # Resize attention map to match image size
    h, w = image.shape[:2]
    att_weight = np.repeat(np.repeat(att_weight, h//att_size, axis=0), w//att_size, axis=1)
    att_weight = att_weight[:h, :w]
    
    # Display image with attention overlay
    plt.imshow(image)
    plt.imshow(att_weight, alpha=0.5, cmap='hot')
    plt.title(f'Attention for: "{word}"', fontsize=FONT_TITLE, pad=10)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=STANDARD_DPI, facecolor='white')
    
    plt.close()

def create_attention_grid(image: np.ndarray, caption: List[str], 
                         attention_weights: List[torch.Tensor], 
                         max_words: int = 12) -> np.ndarray:
    """
    Create a grid of attention visualizations
    
    Args:
        image: Original image
        caption: List of caption words
        attention_weights: List of attention weight tensors
        max_words: Maximum number of words to show
        
    Returns:
        grid_image: Combined grid image
    """
    import cv2
    
    # Limit number of words
    num_words = min(len(caption), max_words)
    
    # Calculate grid dimensions
    grid_cols = 4
    grid_rows = (num_words + grid_cols - 1) // grid_cols
    
    # Resize image for grid
    target_size = (224, 224)
    image_resized = cv2.resize(image, target_size)
    
    # Create grid
    grid_height = grid_rows * target_size[1]
    grid_width = grid_cols * target_size[0]
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    for i in range(num_words):
        row = i // grid_cols
        col = i % grid_cols
        
        # Get attention for this word
        att_weight = attention_weights[i]
        
        # Reshape and resize attention
        att_size = int(np.sqrt(att_weight.shape[0]))
        att_weight = att_weight.reshape(att_size, att_size)
        att_weight = cv2.resize(att_weight, target_size)
        
        # Normalize attention weights
        att_weight = (att_weight - att_weight.min()) / (att_weight.max() - att_weight.min() + 1e-8)
        
        # Create attention overlay
        att_overlay = cv2.applyColorMap((att_weight * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        
        # Blend with original image
        blended = cv2.addWeighted(image_resized, 0.4, att_overlay, 0.6, 0)
        
        # Add word label
        cv2.putText(blended, caption[i], (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Place in grid
        y_start = row * target_size[1]
        y_end = (row + 1) * target_size[1]
        x_start = col * target_size[0]
        x_end = (col + 1) * target_size[0]
        
        grid_image[y_start:y_end, x_start:x_end] = blended
    
    return grid_image

def create_attention_sequence(image: np.ndarray, caption: List[str], 
                            attention_weights: List[torch.Tensor],
                            save_dir: Optional[str] = None,
                            figsize: tuple = STANDARD_FIGSIZE):
    """
    Create individual attention images for each word in sequence
    
    Args:
        image: Original image
        caption: List of caption words
        attention_weights: List of attention weight tensors
        save_dir: Directory to save individual images
        figsize: Figure size for each image
        
    Returns:
        List of file paths if save_dir is provided
    """
    saved_paths = []
    
    for i, (word, att_weight) in enumerate(zip(caption, attention_weights)):
        if save_dir:
            save_path = os.path.join(save_dir, f'attention_{i:02d}_{word}.png')
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_path = None
            
        visualize_single_attention(image, word, att_weight, save_path, figsize)
        
        if save_path:
            saved_paths.append(save_path)
    
    return saved_paths