# src/models/attention.py

"""
Attention-based CNN-LSTM model for image captioning
"""

import torch
import torch.nn as nn
from typing import Tuple, List

from .encoders import EncoderCNN
from ..preprocessing.vocabulary import Vocabulary

class Attention(nn.Module):
    """Attention mechanism for focusing on specific parts of the image"""
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        Initialize attention mechanism
        
        Args:
            encoder_dim: Dimension of encoder output
            decoder_dim: Dimension of decoder hidden state
            attention_dim: Dimension of attention network
        """
        super(Attention, self).__init__()
        
        # Layers for attention mechanism
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Print architecture info
        print(f"Initializing Attention mechanism:")
        print(f"  Encoder dimension: {encoder_dim}")
        print(f"  Decoder dimension: {decoder_dim}")
        print(f"  Attention dimension: {attention_dim}")
        
        # Count parameters
        print(f"  Attention parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the attention layer
        
        Args:
            encoder_out: Feature maps from encoder, shape (batch_size, num_pixels, encoder_dim)
            decoder_hidden: Hidden state of the decoder, shape (batch_size, decoder_dim)
            
        Returns:
            attention_weighted_encoding: Weighted sum of encoder outputs (batch_size, encoder_dim)
            alpha: Attention weights (batch_size, num_pixels)
        """
        # Transform encoder output for attention
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        
        # Transform decoder hidden state for attention
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        
        # Sum and apply non-linearity
        att = self.relu(att1 + att2.unsqueeze(1))  # (batch_size, num_pixels, attention_dim)
        
        # Compute attention scores
        att = self.full_att(att).squeeze(2)  # (batch_size, num_pixels)
        
        # Apply softmax to get attention weights
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        
        # Compute weighted encoding
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        
        return attention_weighted_encoding, alpha

class AttentionDecoderRNN(nn.Module):
    """RNN decoder that uses attention mechanism for generating captions"""
    
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, 
                 encoder_dim: int, attention_dim: int, num_layers: int = 1, 
                 dropout: float = 0.5):
        """
        Initialize attention decoder
        
        Args:
            embed_size: Size of word embeddings
            hidden_size: Size of LSTM hidden state
            vocab_size: Size of vocabulary
            encoder_dim: Dimension of encoder output
            attention_dim: Dimension of attention network
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(AttentionDecoderRNN, self).__init__()
        
        # Store parameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.num_layers = num_layers
        
        # Print architecture information
        print(f"\nInitializing Attention Decoder RNN:")
        print(f"  Embed size: {embed_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Encoder dimension: {encoder_dim}")
        print(f"  Attention dimension: {attention_dim}")
        print(f"  LSTM layers: {num_layers}")
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Attention mechanism
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        
        # Decoder LSTM
        self.lstm = nn.LSTM(
            input_size=embed_size + encoder_dim,  # Input is concat of embedding and context
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer to compute initial hidden/cell states from mean of encoder output
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        
        # Layer to produce word scores
        self.fc = nn.Linear(hidden_size + encoder_dim, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Print parameter counts
        embed_params = sum(p.numel() for p in self.embedding.parameters())
        att_params = sum(p.numel() for p in self.attention.parameters())
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        fc_params = sum(p.numel() for p in self.fc.parameters())
        
        print(f"  Embedding parameters: {embed_params:,}")
        print(f"  Attention parameters: {att_params:,}")
        print(f"  LSTM parameters: {lstm_params:,}")
        print(f"  Output layer parameters: {fc_params:,}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def init_hidden_state(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden and cell states using the encoder output
        
        Args:
            encoder_out: Feature maps from encoder, shape (batch_size, num_pixels, encoder_dim)
            
        Returns:
            h: Initial hidden state (num_layers, batch_size, hidden_size)
            c: Initial cell state (num_layers, batch_size, hidden_size)
        """
        # Mean of encoder output across all pixels
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch_size, encoder_dim)
        
        # Project to get initial states
        h = self.init_h(mean_encoder_out)  # (batch_size, hidden_size)
        c = self.init_c(mean_encoder_out)  # (batch_size, hidden_size)
        
        # Reshape for LSTM which expects (num_layers, batch_size, hidden_size)
        h = h.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = c.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        return h, c
    
    def forward_with_teacher_forcing(self, encoder_out: torch.Tensor, captions: torch.Tensor, 
                                   caption_lengths: List[int], debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with teacher forcing during training
        
        Args:
            encoder_out: Feature maps from encoder, shape (batch_size, encoder_dim, height, width)
            captions: Ground truth captions (batch_size, max_length)
            caption_lengths: True lengths of each caption
            debug: Whether to print debug info
            
        Returns:
            outputs: Predicted word scores (batch_size, max_length, vocab_size)
            alphas: Attention weights for visualization (batch_size, max_length, num_pixels)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(1)
        height = encoder_out.size(2)
        width = encoder_out.size(3)
        max_length = captions.size(1)
        
        # Flatten spatial dimensions of encoder output for attention
        encoder_out = encoder_out.permute(0, 2, 3, 1)  # (batch_size, height, width, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Initialize LSTM hidden and cell states
        h, c = self.init_hidden_state(encoder_out)
        
        # Prepare embeddings for captions
        embeddings = self.dropout(self.embedding(captions))  # (batch_size, max_length, embed_size)
        
        # Initialize tensors for predictions and attention weights
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).to(captions.device)
        alphas = torch.zeros(batch_size, max_length, num_pixels).to(captions.device)
        
        # For each time step
        for t in range(max_length):
            # Get hidden state for attention (using last layer's hidden state)
            h_for_att = h[-1]  # (batch_size, hidden_size)
            
            # Compute attention
            context, alpha = self.attention(encoder_out, h_for_att)
            
            # Store attention weights
            alphas[:, t] = alpha
            
            # Prepare input for LSTM - concatenate context with embedding
            lstm_input = torch.cat([embeddings[:, t], context], dim=1).unsqueeze(1)  # (batch_size, 1, embed_size + encoder_dim)
            
            # Run LSTM step
            output, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Reshape output
            output = output.squeeze(1)  # (batch_size, hidden_size)
            
            # Concatenate output with context for final prediction
            output = torch.cat([output, context], dim=1)  # (batch_size, hidden_size + encoder_dim)
            
            # Generate word scores
            preds = self.fc(self.dropout(output))  # (batch_size, vocab_size)
            
            # Store predictions
            predictions[:, t] = preds
        
        return predictions, alphas
    
    def sample(self, encoder_out: torch.Tensor, max_length: int = 20, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate captions using attention and greedy search
        
        Args:
            encoder_out: Feature maps from encoder (batch_size, encoder_dim, height, width)
            max_length: Maximum caption length
            debug: Whether to print debug info
            
        Returns:
            sampled_ids: Predicted caption indices (batch_size, max_length)
            alphas: Attention weights for visualization (batch_size, max_length, num_pixels)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(1)
        height = encoder_out.size(2)
        width = encoder_out.size(3)
        
        # Flatten spatial dimensions for attention
        encoder_out = encoder_out.permute(0, 2, 3, 1)  # (batch_size, height, width, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Initialize LSTM hidden and cell states
        h, c = self.init_hidden_state(encoder_out)
        
        # Initialize result tensors
        sampled_ids = torch.zeros(batch_size, max_length, dtype=torch.long).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_length, num_pixels).to(encoder_out.device)
        
        # Start with <SOS> token (index 1)
        input_word = torch.ones(batch_size, dtype=torch.long).to(encoder_out.device)
        
        # Generate words one by one
        for t in range(max_length):
            # Embed the input word
            embedded = self.embedding(input_word)  # (batch_size, embed_size)
            
            # Get hidden state for attention
            h_for_att = h[-1]  # (batch_size, hidden_size)
            
            # Compute attention
            context, alpha = self.attention(encoder_out, h_for_att)
            
            # Store attention weights
            alphas[:, t] = alpha
            
            # Prepare input for LSTM
            lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)  # (batch_size, 1, embed_size + encoder_dim)
            
            # Run LSTM step
            output, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Reshape output
            output = output.squeeze(1)  # (batch_size, hidden_size)
            
            # Concatenate output with context
            output = torch.cat([output, context], dim=1)  # (batch_size, hidden_size + encoder_dim)
            
            # Generate word scores
            preds = self.fc(self.dropout(output))  # (batch_size, vocab_size)
            
            # Greedy search - pick highest probability word
            predicted = preds.argmax(dim=1)  # (batch_size)
            
            # Store prediction
            sampled_ids[:, t] = predicted
            
            # Next input is the predicted word
            input_word = predicted
        
        return sampled_ids, alphas

class AttentionCaptionModel(nn.Module):
    """Complete CNN-RNN model with attention for image captioning"""
    
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, 
                 attention_dim: int, num_layers: int = 1, dropout: float = 0.5,
                 train_cnn: bool = False, **encoder_kwargs):
        """
        Initialize attention model
        
        Args:
            embed_size: Size of embeddings
            hidden_size: Size of LSTM hidden state
            vocab_size: Size of vocabulary
            attention_dim: Dimension of attention network
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            train_cnn: Whether to train CNN backbone
            **encoder_kwargs: Additional encoder arguments
        """
        super(AttentionCaptionModel, self).__init__()
        
        # Print architecture information
        print("\nInitializing Attention Caption Model")
        print(f"  Embed size: {embed_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Attention dimension: {attention_dim}")
        print(f"  LSTM layers: {num_layers}")
        
        # Create encoder and decoder
        self.encoder = EncoderCNN(embed_size, dropout, train_cnn=train_cnn, attention=True)
        encoder_dim = embed_size  # The encoder projects to embed_size
        
        self.decoder = AttentionDecoderRNN(
            embed_size, hidden_size, vocab_size, 
            encoder_dim, attention_dim, num_layers, dropout
        )
        
        # Print total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nModel Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor, 
                caption_lengths: List[int], debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training with teacher forcing
        
        Args:
            images: Input images (batch_size, 3, height, width)
            captions: Caption indices (batch_size, max_length)
            caption_lengths: True lengths of captions
            debug: Whether to print debug info
            
        Returns:
            outputs: Predicted word scores (batch_size, max_length, vocab_size)
            alphas: Attention weights (batch_size, max_length, num_pixels)
        """
        # Extract image features
        features = self.encoder(images, debug)  # (batch_size, embed_size, H, W)
        
        # Generate captions with attention
        outputs, alphas = self.decoder.forward_with_teacher_forcing(
            features, captions, caption_lengths, debug)
        
        return outputs, alphas
    
    def sample(self, images: torch.Tensor, max_length: int = 20, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate captions with attention for given images
        
        Args:
            images: Input images (batch_size, 3, height, width)
            max_length: Maximum caption length
            debug: Whether to print debug info
            
        Returns:
            sampled_ids: Generated caption indices (batch_size, max_length)
            alphas: Attention weights for visualization (batch_size, max_length, num_pixels)
        """
        # Extract image features
        features = self.encoder(images, debug)  # (batch_size, embed_size, H, W)
        
        # Generate captions with attention
        sampled_ids, alphas = self.decoder.sample(features, max_length, debug)
        
        return sampled_ids, alphas
    
    def caption_image_with_attention(self, image: torch.Tensor, vocab: Vocabulary, 
                                   max_length: int = 20, debug: bool = False) -> Tuple[str, List[torch.Tensor]]:
        """
        Generate a caption with attention for a single image
        
        Args:
            image: Input image (1, 3, height, width)
            vocab: Vocabulary object
            max_length: Maximum caption length
            debug: Whether to print debug info
            
        Returns:
            caption: Generated caption as string
            attention_weights: Attention weights for visualization
        """
        # Set to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Generate caption indices and attention weights
            sampled_ids, alphas = self.sample(image, max_length, debug)
            
            # Convert indices to words
            sampled_ids = sampled_ids[0].cpu().numpy()
            
            # Create caption
            caption_words = []
            attention_weights = []
            
            for i, idx in enumerate(sampled_ids):
                word = vocab.itos[idx]
                
                # Stop if EOS token
                if word == "<EOS>":
                    attention_weights.append(alphas[0, i].cpu())
                    break
                
                # Skip special tokens
                if word not in ["<PAD>", "<SOS>"]:
                    caption_words.append(word)
                    attention_weights.append(alphas[0, i].cpu())
            
            caption = " ".join(caption_words)
        
        return caption, attention_weights