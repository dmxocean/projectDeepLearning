# src/models/encoders.py

"""
CNN encoders for extracting image features
"""

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """CNN encoder for extracting image features"""
    
    def __init__(self, embed_size: int, dropout: float = 0.5, 
                 train_cnn: bool = False, attention: bool = False):
        """
        Initialize encoder
        
        Args:
            embed_size: Size of embedding dimension
            dropout: Dropout probability
            train_cnn: Whether to train CNN backbone
            attention: Whether this is for attention model (preserves spatial info)
        """
        super(EncoderCNN, self).__init__()
        
        # Load pre-trained ResNet-50
        resnet = models.resnet50(pretrained=True)
        
        # Different handling for attention vs baseline
        if attention:
            # For attention: Keep spatial information, remove final FC and pooling
            modules = list(resnet.children())[:-2]
            print(f"Initializing Encoder CNN with spatial features:")
            self.feature_size = 2048  # ResNet features without pooling
            
            # Conv layer to reduce channel dimension
            self.conv = nn.Conv2d(self.feature_size, embed_size, kernel_size=1)
            
        else:
            # For baseline: Use pooled features
            modules = list(resnet.children())[:-1]
            print(f"Initializing Encoder CNN:")
            # Save the feature size
            self.feature_size = resnet.fc.in_features
            
            # Project to embedding space
            self.fc = nn.Linear(self.feature_size, embed_size)
            
            # Use LayerNorm instead of BatchNorm (works with small batches)
            self.norm = nn.LayerNorm(embed_size)
            
        # Create resnet feature extractor
        self.resnet = nn.Sequential(*modules)
        self.attention = attention
        
        # Freeze or unfreeze CNN
        for param in self.resnet.parameters():
            param.requires_grad = train_cnn
        
        # Additional layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Print architecture info
        print(f"  Embed size: {embed_size}")
        print(f"  Feature size: {self.feature_size}")
        print(f"  Using attention: {attention}")
        print(f"  Training CNN backbone: {train_cnn}")
        
        # Count parameters
        if attention:
            print(f"  Conv parameters: {sum(p.numel() for p in self.conv.parameters()):,}")
        else:
            print(f"  FC parameters: {sum(p.numel() for p in self.fc.parameters()):,}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, images: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Extract features from images
        
        Args:
            images: Input images of shape (batch_size, 3, 224, 224)
            debug: Whether to print debug info
            
        Returns:
            features: For baseline: (batch_size, embed_size)
                     For attention: (batch_size, embed_size, H, W) where H=W=7 for 224x224 input
        """
        # Get features from ResNet
        features = self.resnet(images)
        # features shape: (batch_size, 2048, H, W) for attention or (batch_size, 2048, 1, 1) for baseline
        
        # Different processing for attention vs baseline
        if self.attention:
            # For attention model: Use 1x1 conv to reduce channels
            features = self.conv(features)  # (batch_size, embed_size, H, W)
            
            # Apply ReLU and dropout
            features = self.dropout(self.relu(features))
            
        else:
            # For baseline model: Flatten and project
            # Reshape: (batch_size, 2048, 1, 1) -> (batch_size, 2048)
            features = features.view(features.size(0), -1)
            
            # Project to embedding space
            features = self.fc(features)  # (batch_size, embed_size)
            
            # Apply normalization, dropout and ReLU
            features = self.norm(features)
            features = self.dropout(self.relu(features))
        
        return features