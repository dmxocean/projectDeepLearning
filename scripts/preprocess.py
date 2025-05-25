# scripts/preprocess.py

"""
Data preprocessing script for image captioning
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project modules
from src.utils.manager import ConfigManager
from src.utils.constants import SEED
from src.utils.io import save_pickle, ensure_dir, save_json
from src.preprocessing.vocabulary import Vocabulary, preprocess_caption, analyze_vocab_coverage
from src.preprocessing.dataset import create_data_splits

def main():
    """Main preprocessing function"""
    
    # Set random seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Get configurations
    data_config = config_manager.get_data_params()
    debug_mode = config_manager.debug
    
    print(f"Debug mode: {debug_mode}")
    if debug_mode:
        print(f"Max images in debug mode: {data_config['debug']['max_images']}")
        print(f"Output directory: {config_manager.paths['processed']}")
    
    # Ensure output directories exist
    ensure_dir(config_manager.paths['processed'])
    
    # Load captions
    captions_file = data_config['dataset']['captions_file']
    images_dir = data_config['dataset']['images_dir']
    
    print(f"Loading captions from: {captions_file}")
    captions_df = pd.read_csv(captions_file)
    print(f"Loaded {len(captions_df)} captions")
    
    # If debug mode, limit dataset
    if debug_mode:
        max_images = data_config['debug']['max_images']
        unique_images = captions_df['image'].unique()[:max_images]
        captions_df = captions_df[captions_df['image'].isin(unique_images)].reset_index(drop=True)
        print(f"\nDEBUG MODE: Limited to {len(unique_images)} images, {len(captions_df)} captions")
    
    # Process captions
    print("\nProcessing captions...")
    captions_df['processed_caption'] = captions_df['caption'].apply(preprocess_caption)
    
    # Show examples
    print("\nSample processed captions:")
    for i in range(min(3, len(captions_df))):
        print(f"Original: {captions_df.iloc[i]['caption']}")
        print(f"Processed: {captions_df.iloc[i]['processed_caption']}")
        print()
    
    # Caption length analysis
    captions_df['caption_length'] = captions_df['processed_caption'].apply(lambda x: len(x.split()))
    print(f"Caption length statistics:")
    print(captions_df['caption_length'].describe())
    
    # Create splits
    train_ratio = data_config['preprocessing']['train_split']
    val_ratio = data_config['preprocessing']['val_split']
    test_ratio = data_config['preprocessing']['test_split']
    
    print(f"\nCreating splits: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    train_df, val_df, test_df = create_data_splits(
        captions_df, 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=SEED
    )
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} captions, {len(train_df['image'].unique())} images")
    print(f"Val: {len(val_df)} captions, {len(val_df['image'].unique())} images")
    print(f"Test: {len(test_df)} captions, {len(test_df['image'].unique())} images")
    
    # Verify no overlap between splits
    train_images = set(train_df['image'].unique())
    val_images = set(val_df['image'].unique())
    test_images = set(test_df['image'].unique())
    
    print("\nChecking for overlaps between splits:")
    print(f"Train-Val overlap: {len(train_images & val_images)} images")
    print(f"Train-Test overlap: {len(train_images & test_images)} images")
    print(f"Val-Test overlap: {len(val_images & test_images)} images")
    
    assert len(train_images & val_images) == 0, "Train and validation sets overlap!"
    assert len(train_images & test_images) == 0, "Train and test sets overlap!"
    assert len(val_images & test_images) == 0, "Validation and test sets overlap!"
    print("No overlaps found!")
    
    # Build vocabulary from training set only
    vocab_threshold = data_config['preprocessing']['vocab_threshold']
    print(f"\nBuilding vocabulary with frequency threshold: {vocab_threshold}")
    
    vocab = Vocabulary(freq_threshold=vocab_threshold)
    vocab.build_vocabulary(train_df['processed_caption'].tolist())
    
    # Show vocabulary statistics
    print(f"\nVocabulary statistics:")
    print(f"Total unique words seen: {len(vocab.word_frequencies)}")
    print(f"Words in vocabulary: {len(vocab) - 4}")
    print(f"Total vocabulary size (with special tokens): {len(vocab)}")
    
    # Analyze vocabulary coverage
    print("\nVocabulary coverage analysis:")
    print("\nTraining set:")
    train_coverage, _ = analyze_vocab_coverage(train_df, vocab)
    
    print("\nValidation set:")
    val_coverage, _ = analyze_vocab_coverage(val_df, vocab)
    
    print("\nTest set:")
    test_coverage, _ = analyze_vocab_coverage(test_df, vocab)
    
    print(f"\nSummary:")
    print(f"Train coverage: {train_coverage:.2f}%")
    print(f"Val coverage: {val_coverage:.2f}%")
    print(f"Test coverage: {test_coverage:.2f}%")
    
    # Show most frequent words
    print("\nMost frequent words in vocabulary:")
    most_freq = vocab.get_most_frequent_words(20)
    for i, (word, count) in enumerate(most_freq[:20], 1):
        print(f"{i:2d}. '{word}': {count} times")
    
    # Save vocabulary
    vocab_path = config_manager.paths['vocab']
    vocab.save(vocab_path)
    print(f"\nSaved vocabulary to: {vocab_path}")
    
    # Save data splits
    splits_path = config_manager.paths['splits']
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    save_pickle(splits, splits_path)
    print(f"Saved data splits to: {splits_path}")
    
    # Save summary
    summary = {
        'dataset': data_config['dataset']['name'],
        'debug_mode': debug_mode,
        'vocab_size': len(vocab),
        'vocab_threshold': vocab_threshold,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_images': len(train_df['image'].unique()),
        'val_images': len(val_df['image'].unique()),
        'test_images': len(test_df['image'].unique()),
        'train_coverage': train_coverage,
        'val_coverage': val_coverage,
        'test_coverage': test_coverage
    }
    
    summary_path = os.path.join(config_manager.paths['processed'], 'preprocessing_summary.json')
    save_json(summary, summary_path)
    print(f"Saved preprocessing summary to: {summary_path}")
    
    # Print final summary
    print("\nPREPROCESSING COMPLETE")
    print()
    print(f"Debug mode: {debug_mode}")
    print(f"Output directory: {config_manager.paths['processed']}")
    print(f"\nDataset:")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"\nCoverage:")
    print(f"  Train: {train_coverage:.2f}%")
    print(f"  Val: {val_coverage:.2f}%")
    print(f"  Test: {test_coverage:.2f}%")
    print(f"\nFiles saved:")
    print(f"  - {vocab_path}")
    print(f"  - {splits_path}")
    print(f"  - {summary_path}")

if __name__ == "__main__":
    main()