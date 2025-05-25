# src/preprocessing/transforms.py

"""
Image transformation pipelines for model training and evaluation
"""

import torch
import numpy as np
import torchvision.transforms as transforms
from typing import Tuple
from ..utils.constants import IMAGENET_MEAN, IMAGENET_STD

def get_transforms(resize: int = 256, crop: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create image transformations for training and validation/test sets
    
    Args:
        resize: Size to resize images to
        crop: Size to crop images to
        
    Returns:
        transform_train: Transformations for training
        transform_val: Transformations for validation/testing
    """
    # Training transforms with data augmentation
    transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Validation/test transforms (no augmentation)
    transform_val = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return transform_train, transform_val

def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image using ImageNet statistics
    
    Args:
        image: Image tensor of shape (3, H, W) or (B, 3, H, W)
        
    Returns:
        normalized: Normalized image tensor
    """
    # Create transform
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    # Apply normalization
    if image.dim() == 3:  # Single image
        return normalize(image)
    else:  # Batch of images
        return torch.stack([normalize(img) for img in image])

def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization
    
    Args:
        image_tensor: Normalized image tensor
        
    Returns:
        image: Denormalized image as numpy array
    """
    # Convert to numpy and move channels to the end
    if image_tensor.dim() == 4:  # Batch of images
        image = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    else:  # Single image
        image = image_tensor.cpu().permute(1, 2, 0).numpy()
    
    # Reverse normalization
    image = image * np.array(IMAGENET_STD).reshape(1, 1, 3) + np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    
    # Clip values
    image = np.clip(image, 0, 1)
    return image