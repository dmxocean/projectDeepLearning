# src/models/baseline.py

"""
Baseline CNN-LSTM model for image captioning
"""

import torch
import torch.nn as nn
from typing import Tuple, List

from .encoders import EncoderCNN
from ..preprocessing.vocabulary import Vocabulary

class DecoderRNN(nn.Module):
    """RNN decoder for generating captions"""
    
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, 
                 num_layers: int = 1, dropout: float = 0.5):
        """
        Initialize decoder
        
        Args:
            embed_size: Size of word embeddings
            hidden_size: Size of LSTM hidden state
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(DecoderRNN, self).__init__()
        
        # Store parameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Print architecture information
        print(f"\nInitializing Decoder RNN:")
        print(f"  Embed size: {embed_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  LSTM layers: {num_layers}")
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Print parameter counts
        embed_params = sum(p.numel() for p in self.embedding.parameters())
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        output_params = sum(p.numel() for p in self.output_layer.parameters())
        
        print(f"  Embedding parameters: {embed_params:,}")
        print(f"  LSTM parameters: {lstm_params:,}")
        print(f"  Output layer parameters: {output_params:,}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward_with_teacher_forcing(self, features: torch.Tensor, captions: torch.Tensor, 
                                   caption_lengths: List[int], debug: bool = False) -> torch.Tensor:
        """
        Forward pass with teacher forcing during training
        
        Args:
            features: Image features from encoder (batch_size, embed_size)
            captions: Ground truth captions (batch_size, max_length)
            caption_lengths: True lengths of each caption
            debug: Whether to print debug info
            
        Returns:
            outputs: Predicted word scores (batch_size, max_length, vocab_size)
        """
        batch_size = features.size(0)
        max_length = captions.size(1)
        
        # Prepare embeddings for captions
        embeddings = self.dropout(self.embedding(captions))  # (batch_size, max_length, embed_size)
        
        # Prepare to include features as first input
        # Reshape features: (batch_size, embed_size) -> (batch_size, 1, embed_size)
        features = features.unsqueeze(1)
        
        # For teacher forcing, we'll use features for the first time step,
        # then the embeddings of the ground truth tokens
        decoder_input = torch.cat([features, embeddings[:, :-1, :]], dim=1)  # (batch_size, max_length, embed_size)
        
        # Run through LSTM
        outputs, _ = self.lstm(decoder_input)  # (batch_size, max_length, hidden_size)
        
        # Generate word scores
        outputs = self.output_layer(outputs)  # (batch_size, max_length, vocab_size)
        
        return outputs
    
    def sample(self, features: torch.Tensor, max_length: int = 20, debug: bool = False) -> torch.Tensor:
        """
        Generate captions using greedy search
        
        Args:
            features: Image features from encoder (batch_size, embed_size)
            max_length: Maximum caption length
            debug: Whether to print debug info
            
        Returns:
            sampled_ids: Predicted caption indices (batch_size, max_length)
        """
        batch_size = features.size(0)
        
        # Initialize result tensor
        sampled_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=features.device)
        
        # Initialize hidden and cell states
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device)
        
        # First input is the image features
        input_word = features
        
        # Generate words one by one
        for i in range(max_length):
            # Run LSTM step
            output, (h, c) = self.lstm(input_word.unsqueeze(1), (h, c))  # output: (batch_size, 1, hidden_size)
            
            # Get word predictions
            output = self.output_layer(output.squeeze(1))  # (batch_size, vocab_size)
            
            # Greedy search - pick highest probability word
            predicted = output.argmax(dim=1)  # (batch_size,)
            
            # Save prediction
            sampled_ids[:, i] = predicted
            
            # Next input is the predicted word embedding
            input_word = self.embedding(predicted)  # (batch_size, embed_size)
        
        return sampled_ids

class BaselineCaptionModel(nn.Module):
    """Complete CNN-RNN model for image captioning"""
    
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, 
                 num_layers: int = 1, dropout: float = 0.5,
                 train_cnn: bool = False, **encoder_kwargs):
        """
        Initialize baseline model
        
        Args:
            embed_size: Size of embeddings
            hidden_size: Size of LSTM hidden state
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            train_cnn: Whether to train CNN backbone
            **encoder_kwargs: Additional encoder arguments
        """
        super(BaselineCaptionModel, self).__init__()
        
        # Print architecture information
        print("\nInitializing Baseline Caption Model")
        print(f"  Embed size: {embed_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  LSTM layers: {num_layers}")
        
        # Create encoder and decoder
        self.encoder = EncoderCNN(embed_size, dropout, train_cnn=train_cnn, attention=False)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, dropout)
        
        # Print total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nModel Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor, 
                caption_lengths: List[int], debug: bool = False) -> torch.Tensor:
        """
        Forward pass for training
        
        Args:
            images: Input images (batch_size, 3, height, width)
            captions: Caption indices (batch_size, max_length)
            caption_lengths: True lengths of captions
            debug: Whether to print debug info
            
        Returns:
            outputs: Predicted word scores (batch_size, max_length, vocab_size)
        """
        # Extract image features
        features = self.encoder(images, debug)  # (batch_size, embed_size)
        
        # Generate captions
        outputs = self.decoder.forward_with_teacher_forcing(
            features, captions, caption_lengths, debug)  # (batch_size, max_length, vocab_size)
        
        return outputs
    
    def sample(self, images: torch.Tensor, max_length: int = 20, debug: bool = False) -> torch.Tensor:
        """
        Generate captions for given images
        
        Args:
            images: Input images (batch_size, 3, height, width)
            max_length: Maximum caption length
            debug: Whether to print debug info
            
        Returns:
            sampled_ids: Generated caption indices (batch_size, max_length)
        """
        # Extract image features
        features = self.encoder(images, debug)  # (batch_size, embed_size)
        
        # Generate captions
        sampled_ids = self.decoder.sample(features, max_length, debug)  # (batch_size, max_length)
        
        return sampled_ids
    
    def caption_image(self, image: torch.Tensor, vocab: Vocabulary, 
                     max_length: int = 20, debug: bool = False) -> str:
        """
        Generate a caption for a single image
        
        Args:
            image: Input image (1, 3, height, width)
            vocab: Vocabulary object
            max_length: Maximum caption length
            debug: Whether to print debug info
            
        Returns:
            caption: Generated caption as string
        """
        # Set to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Generate caption indices
            sampled_ids = self.sample(image, max_length, debug)  # (1, max_length)
            
            # Convert indices to words
            sampled_ids = sampled_ids[0].cpu().numpy()
            
            # Create caption
            caption_words = []
            for idx in sampled_ids:
                word = vocab.itos[idx]
                
                # Stop if EOS token
                if word == "<EOS>":
                    break
                
                # Skip special tokens
                if word not in ["<PAD>", "<SOS>"]:
                    caption_words.append(word)
            
            caption = " ".join(caption_words)
        
        return caption