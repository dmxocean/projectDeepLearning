# Model Architecture Specifications

## Abstract
This document provides detailed technical specifications for the two neural architectures implemented in this image captioning system. 

- The baseline CNN-LSTM model establishes a performance benchmark (`BLEU-1: 21.77%`)
- The attention-enhanced model demonstrates superior performance (`BLEU-1: 35.62%`) through dynamic visual focus mechanisms

## 1. Baseline CNN-LSTM Architecture

### 1.1 Architecture Overview
The baseline model implements a traditional encoder-decoder framework with global image representation:

```
+---------------------------------------------------------------------+
|                    Baseline CNN-LSTM Architecture                   |
+---------------------------------------------------------------------+
|                                                                     |
|  Input Image                                                        |
|   224x224x3                                                         |
|       |                                                             |
|       v                                                             |
|  +-------------+                                                    |
|  |  ResNet-50  |  --> Global Average Pooling                        |
|  |   Encoder   |      2048 dimensions                               |
|  +-------------+                                                    |
|       |                                                             |
|       v                                                             |
|  +-------------+                                                    |
|  |   Linear    |  --> Feature Projection                            |
|  | 2048 -> 512 |      512 dimensions                                |
|  +-------------+                                                    |
|       |                                                             |
|       v                                                             |
|  +-------------+     +-------------+     +-------------+            |
|  |    Word     |---->|    LSTM     |---->|   Linear    |            |
|  |  Embedding  |     |   Decoder   |     |  Classifier |            |
|  |   256-dim   |     |  512 units  |     |  Vocab Size |            |
|  +-------------+     +-------------+     +-------------+            |
|                                                 |                   |
|                                                 v                   |
|                                           Caption Output            |
|                                                                     |
+---------------------------------------------------------------------+
```

### 1.2 Component Specifications

| Component | Input Dim | Output Dim | Parameters | Description |
|-----------|-----------|------------|------------|-------------|
| ResNet-50 Encoder | `224x224x3` | `2048` | `23.51M` | Pre-trained feature extractor |
| Feature Projection | `2048` | `512` | `1.05M` | Linear transformation |
| Word Embedding | `Vocab` | `256` | `0.65M` | Learnable word representations |
| LSTM Decoder | `768` | `512` | `2.36M` | Sequential caption generation |
| Output Classifier | `512` | `Vocab` | `1.30M` | Probability distribution |
| **Total** | - | - | **`27.54M`** | **`4.03M` trainable** |

### 1.3 Information Flow
The baseline model processes information through the following pipeline:

1. **Image Encoding**: `ResNet-50` extracts global features via average pooling
2. **Feature Projection**: Linear layer reduces dimensionality to match decoder
3. **Hidden State Init**: Image features initialize LSTM hidden state
4. **Sequential Decoding**: LSTM generates words using teacher forcing
5. **Output Generation**: Linear classifier produces vocabulary probabilities

**Key Limitation**: Information bottleneck at global pooling stage loses spatial details - this is where we really see the difference!

## 2. Attention-Enhanced CNN-LSTM Architecture

### 2.1 Architecture Overview
The attention model preserves spatial information and implements dynamic focus:

```
+---------------------------------------------------------------------+
|                Attention-Enhanced CNN-LSTM Architecture             |
+---------------------------------------------------------------------+
|                                                                     |
|  Input Image                                                        |
|   224x224x3                                                         |
|       |                                                             |
|       v                                                             |
|  +-------------+                                                    |
|  |  ResNet-50  |  --> Spatial Feature Maps                          |
|  |   Encoder   |      7x7x2048 dimensions                           |
|  +-------------+                                                    |
|       |                                                             |
|       v                                                             |
|  +-------------+                                                    |
|  |   Linear    |  --> Feature Projection                            |
|  | 2048 -> 512 |      7x7x512 dimensions                            |
|  +-------------+                                                    |
|       |                                                             |
|       v                                                             |
|  +---------------------------------------------------------+        |
|  |                Attention Mechanism                      |        |
|  |                                                         |        |
|  |  +-----------+    +-------------+    +-------------+    |        |
|  |  |  Hidden   |--->|   Attention |--->|   Context   |    |        |
|  |  |   State   |    |   Weights   |    |   Vector    |    |        |
|  |  |   h(t-1)  |    |  alpha(t,i) |    |    c(t)     |    |        |
|  |  +-----------+    +-------------+    +-------------+    |        |
|  +---------------------------------------------------------+        |
|                               |                                     |
|                               v                                     |
|  +-------------+     +-------------+     +-------------+            |
|  |    Word     |---->|    LSTM     |---->|   Linear    |            |
|  |  Embedding  |     |   Decoder   |     |  Classifier |            |
|  |   256-dim   |     |  512 units  |     | Vocab Size  |            |
|  +-------------+     +-------------+     +-------------+            |
|                                                 |                   |
|                                                 v                   |
|                                           Caption Output            |
|                                                                     |
+---------------------------------------------------------------------+
```

### 2.2 Component Specifications

| Component | Input Dim | Output Dim | Parameters | Description |
|-----------|-----------|------------|------------|-------------|
| ResNet-50 Encoder | `224x224x3` | `7x7x2048` | `23.51M` | Spatial feature extraction |
| Feature Projection | `2048` | `512` | `1.05M` | Preserve spatial dimensions |
| Attention Mechanism | `512+512` | `49` | `0.39M` | Bahdanau attention |
| Word Embedding | `Vocab` | `256` | `0.65M` | Learnable representations |
| LSTM Decoder | `768` | `512` | `2.36M` | Context-aware generation |
| Output Classifier | `512` | `Vocab` | `1.30M` | Vocabulary probabilities |
| **Total** | - | - | **`29.17M`** | **`5.66M` trainable** |

### 2.3 Attention Mechanism Details
The attention mechanism computes relevance weights using additive scoring - here's how it works:

**Attention Score Computation**:
$e_{t,i} = W_a \cdot \tanh(W_h h_{t-1} + W_v v_i)$

**Attention Weight Normalization**:
$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{49} \exp(e_{t,j})}$

**Context Vector Generation**:
$c_t = \sum_{i=1}^{49} \alpha_{t,i} v_i$

Where:
- $h_{t-1}$: Previous hidden state (`512-dim`)
- $v_i$: Feature vector at spatial location `i` (`512-dim`)
- $\alpha_{t,i}$: Attention weight for location `i` at time `t`
- $c_t$: Context vector at time `t` (`512-dim`)

### 2.4 Attention Dimensions

| Layer | Weight Matrix | Dimensions | Purpose |
|-------|---------------|------------|---------|
| Hidden Projection | $W_h$ | `256x512` | Process LSTM state |
| Visual Projection | $W_v$ | `256x512` | Process image features |
| Attention Scorer | $W_a$ | `1x256` | Compute attention scores |

## 3. Architecture Comparison

### 3.1 Key Differences

| Aspect | Baseline Model | Attention Model |
|--------|----------------|-----------------|
| **Image Representation** | Global vector (`2048-d`) | Spatial grid (`7x7x512`) |
| **Information Bottleneck** | Severe (single vector) | Minimal (preserved spatial) |
| **Decoder Input** | Image features + embedding | Context + embedding |
| **Computational Cost** | Lower | `5%` increase |
| **Interpretability** | Limited | High (attention weights) |

### 3.2 Performance Impact

| Metric | Baseline | Attention | Improvement |
|--------|----------|-----------|-------------|
| BLEU-1 | `21.77%` | `35.62%` | **+63.6%** |
| BLEU-4 | `0.00%` | `6.19%` | **infinite** |
| Parameters | `27.54M` | `29.17M` | `+5.9%` |
| Training Time | `73 min` | `77 min` | `+5.5%` |

### 3.3 Architectural Advantages

**Baseline Strengths**:
- Computational efficiency
- Simple implementation
- Fast inference (`45ms/image`)
- Lower memory requirements

**Attention Advantages**:
- Spatial awareness preservation
- Dynamic focus capability
- Better handling of complex scenes
- Interpretable decision process
- Superior caption quality

## 4. Implementation Notes

### 4.1 Key Design Decisions
- **ResNet-50 Choice**: Balance between performance and computational cost
- **Attention Dimension**: `256-d` provides optimal trade-off
- **LSTM Size**: `512` units sufficient for vocabulary complexity
- **Embedding Size**: `256-d` captures semantic relationships

### 4.2 Training Considerations
- **Teacher Forcing**: Used during training for both models
- **Gradient Clipping**: Norm `3.0` prevents exploding gradients  
- **Dropout**: `0.5` applied to embeddings and LSTM outputs
- **Learning Rate**: $3 \times 10^{-4}$ with adaptive scheduling