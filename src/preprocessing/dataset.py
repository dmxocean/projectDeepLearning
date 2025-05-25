# src/preprocessing/dataset.py

"""
Dataset and dataloader implementations for image captioning
"""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List
import numpy as np
import pandas as pd

from .vocabulary import Vocabulary
from .transforms import get_transforms
from ..utils.constants import PAD_IDX, SOS_IDX, EOS_IDX

class FlickrDataset(Dataset):
    """Dataset class for Flickr8k images and captions"""
    
    def __init__(self, data_df: pd.DataFrame, root_dir: str, vocab: Vocabulary, 
                 transform=None, caption_col: str = 'processed_caption'):
        """
        Initialize the dataset
        
        Args:
            data_df: DataFrame containing image filenames and captions
            root_dir: Directory containing the images
            vocab: Vocabulary object for processing captions
            transform: Optional image transformations
            caption_col: Column name for captions in data_df
        """
        self.data_df = data_df
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform
        self.caption_col = caption_col
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an image and its corresponding caption
        
        Returns:
            image: Image tensor of shape (3, H, W)
            caption: Caption indices tensor of shape (seq_length,)
        """
        # Get caption and image path
        caption = self.data_df.iloc[idx][self.caption_col]
        img_name = self.data_df.iloc[idx]['image']
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)
        
        # Process caption: convert to indices
        caption_tokens = [SOS_IDX]  # Start with SOS token
        caption_tokens.extend(self.vocab.numericalize(caption))
        caption_tokens.append(EOS_IDX)  # End with EOS token
        
        # Convert to tensor
        caption_tensor = torch.tensor(caption_tokens)
        
        return image, caption_tensor

class FlickrCollate:
    """Custom collate function to handle variable-length captions"""
    
    def __init__(self, pad_idx: int = PAD_IDX):
        self.pad_idx = pad_idx
    
    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Args:
            batch: List of tuples (image, caption)
            
        Returns:
            images: Tensor of shape (batch_size, 3, height, width)
            captions: Padded tensor of shape (batch_size, max_length)
            caption_lengths: List of caption lengths
        """
        # Sort batch by caption length (descending) for packing
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Separate images and captions
        images, captions = zip(*batch)
        
        # Stack images
        images = torch.stack(images, dim=0)
        
        # Get caption lengths
        caption_lengths = [len(cap) for cap in captions]
        
        # Pad captions to have same length
        captions_padded = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        
        return images, captions_padded, caption_lengths

def create_data_splits(df: pd.DataFrame, train_ratio: float = 0.7, 
                      val_ratio: float = 0.15, test_ratio: float = 0.15, 
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split images into train, validation, and test sets using stratified sampling
    
    Args:
        df: DataFrame containing image filenames and captions
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_state: Random seed
        
    Returns:
        train_df, val_df, test_df: DataFrames for each split
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Get unique image IDs
    unique_images = df['image'].unique()
    
    # Create stratification features for better distribution
    strat_features = {}
    
    # Caption length feature (short, medium, long)
    caption_lengths = {}
    for img in unique_images:
        img_captions = df[df['image'] == img]['processed_caption']
        avg_len = sum(len(cap.split()) for cap in img_captions) / len(img_captions)
        caption_lengths[img] = avg_len
    
    # Determine caption length categories
    caption_lens = np.array(list(caption_lengths.values()))
    q1, q2 = np.percentile(caption_lens, [33, 66])
    
    for img, length in caption_lengths.items():
        if length <= q1:
            strat_features[img] = 'short'
        elif length <= q2:
            strat_features[img] = 'medium'
        else:
            strat_features[img] = 'long'
    
    # Create arrays for stratified split
    image_array = np.array(list(strat_features.keys()))
    strat_array = np.array([strat_features[img] for img in image_array])
    
    # Use stratified split
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_imgs, test_imgs, _, _ = train_test_split(
        image_array, strat_array,
        test_size=test_ratio,
        random_state=random_state,
        stratify=strat_array
    )
    
    # Second split: train vs val
    # Recalculate stratification features for the train+val set
    strat_array_train_val = np.array([strat_features[img] for img in train_val_imgs])
    
    # Split train+val into train and val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_imgs, val_imgs, _, _ = train_test_split(
        train_val_imgs, strat_array_train_val,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=strat_array_train_val
    )
    
    # Create dataframes for each split
    train_df = df[df['image'].isin(train_imgs)].reset_index(drop=True)
    val_df = df[df['image'].isin(val_imgs)].reset_index(drop=True)
    test_df = df[df['image'].isin(test_imgs)].reset_index(drop=True)
    
    # Verify stratification worked
    print("Stratification verification:")
    for caption_type in ['short', 'medium', 'long']:
        train_count = sum(1 for img in train_imgs if strat_features[img] == caption_type)
        val_count = sum(1 for img in val_imgs if strat_features[img] == caption_type)
        test_count = sum(1 for img in test_imgs if strat_features[img] == caption_type)
        
        print(f"  {caption_type.capitalize()} captions - "
              f"Train: {train_count/len(train_imgs)*100:.1f}%, "
              f"Val: {val_count/len(val_imgs)*100:.1f}%, "
              f"Test: {test_count/len(test_imgs)*100:.1f}%")
    
    return train_df, val_df, test_df

def create_data_loaders(train_dataset: Dataset, val_dataset: Dataset, 
                       test_dataset: Dataset, batch_size: int, 
                       vocab: Vocabulary) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for all splits
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        vocab: Vocabulary object
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders
    """
    pad_idx = vocab.stoi["<PAD>"]
    
    # Worker settings
    import sys
    num_workers = 4 if sys.platform != 'win32' else 0
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=FlickrCollate(pad_idx=pad_idx),
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=FlickrCollate(pad_idx=pad_idx),
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=FlickrCollate(pad_idx=pad_idx),
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader

def create_debug_loader(dataset: Dataset, batch_size: int = 8, 
                       num_samples: int = 100, vocab: Vocabulary = None) -> DataLoader:
    """Create a smaller loader for debugging"""
    indices = list(range(min(num_samples, len(dataset))))
    subset = Subset(dataset, indices)
    
    pad_idx = vocab.stoi["<PAD>"]
    
    loader = DataLoader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use single process
        collate_fn=FlickrCollate(pad_idx=pad_idx),
        pin_memory=False
    )
    
    return loader