# Image Captioning with Attention Mechanism

## Abstract

This project presents a comprehensive implementation and comparative analysis of deep learning architectures for automatic image captioning. We investigate two primary approaches:

* Baseline `CNN-LSTM` encoder-decoder architecture
* Enhanced model incorporating `Bahdanau-style attention mechanisms`

Our implementation leverages the `Flickr8k` dataset, containing `8,000` images with five human-annotated captions each, to evaluate the effectiveness of attention mechanisms in generating contextually relevant and grammatically coherent image descriptions. Experimental results demonstrate that the attention-based model achieves superior performance across all BLEU metrics, with particularly dramatic improvements in higher-order n-gram matching (`BLEU-4: 0.00%` to `6.19%`).

## Project Structure

```
.
├── config/                    # Configuration specifications
│   ├── data.yaml              
│   ├── model.yaml             
│   ├── training.yaml          
│   └── wandb.yaml             
│
├── data/                      # Dataset repository
│   └── raw/                   # Flickr8k images and annotations
│
├── scripts/                   # Training executables
│   ├── baseline.py            # Baseline model training
│   └── attention.py           # Attention model training
│
├── src/                       # Core implementation
│   ├── models/                # Neural architectures
│   ├── preprocessing/         # Data pipeline
│   ├── training/              # Training framework
│   ├── comparison/            # Evaluation tools
│   ├── visualization/         # Attention visualization
│   └── utils/                 # Utility functions
│
├── notebooks/                 # Analysis notebooks
└── environment.yml            # Conda environment specification
```

## 1. Introduction

Image captioning represents a fundamental challenge at the intersection of computer vision and natural language processing, requiring models to understand visual content and generate linguistically coherent descriptions. 

This project implements and analyzes two neural architectures that address this challenge through different mechanisms of visual-linguistic alignment.

The primary objective is to demonstrate how attention mechanisms enhance the caption generation process by allowing the model to dynamically focus on relevant image regions when generating each word, compared to a baseline approach that encodes the entire image into a fixed representation.

## 2. Methodology

### 2.1 Dataset

The `Flickr8k` dataset provides a controlled environment for image captioning research, containing:

* `8,000` images depicting everyday scenes and activities
* `40,000` captions (5 per image) providing diverse linguistic descriptions
* Balanced representation of objects, people, animals, and scenes
* Average caption length of `11.8 words`

### 2.2 Architectural Approaches

#### Baseline CNN-LSTM Architecture

The baseline model employs a traditional encoder-decoder framework where a convolutional neural network extracts global image features, which are then decoded by a recurrent neural network into a caption sequence. The entire image is compressed into a fixed `2048-dimensional` vector, creating an information bottleneck that significantly impacts performance.

#### Attention-Enhanced Architecture

The attention model extends the baseline by preserving spatial information from the CNN encoder (`7×7×2048 feature maps`) and implementing a dynamic attention mechanism that computes relevance weights for different image regions at each decoding step. This allows the model to selectively focus on different parts of the image when generating each word.

### 2.3 Training Methodology

Both models are trained using teacher forcing with cross-entropy loss optimization. The attention model incorporates additional regularization through doubly stochastic attention constraints to encourage comprehensive image coverage during caption generation.

## 3. Implementation Details

### 3.1 Preprocessing Pipeline

* **Visual Processing**: Images are resized to `224×224` pixels and normalized using ImageNet statistics
* **Linguistic Processing**: Captions are tokenized and vocabulary is constructed with frequency thresholding (minimum `5` occurrences)
* **Data Augmentation**: Training images undergo random transformations to improve generalization

### 3.2 Model Specifications

* **Encoder**: `ResNet-50` pretrained on ImageNet, adapted for feature extraction
* **Decoder**: `LSTM` with `512` hidden units and `256-dimensional` word embeddings
* **Attention**: `256-dimensional` attention space with additive (Bahdanau) scoring
* **Vocabulary**: `2,538` unique tokens including special tokens

### 3.3 Optimization Strategy

* **Optimizer**: Adam with initial learning rate of $3 \times 10^{-4}$
* **Scheduling**: `ReduceLROnPlateau` with patience of `3 epochs`
* **Regularization**: Dropout (`0.5`), gradient clipping (norm `3.0`), and early stopping

## 4. Results and Analysis

### 4.1 Quantitative Evaluation

| Model Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Val Loss | Parameters |
|-------------------|--------|--------|--------|--------|----------|------------|
| Baseline CNN-LSTM | `21.77%` | `5.50%`  | `0.39%`  | `0.00%`  | `3.548`    | `27.54M`     |
| Attention CNN-LSTM| `35.62%` | `18.34%` | `10.63%` | `6.19%`  | `2.858`    | `29.17M`     |
| **Improvement**   | **+63.6%** | **+233.5%**| **+2627%** | **infinite** | **-19.5%**   | **+5.9%**      |

![Baseline Training Curves](https://github.com/ML-DL-Teaching/deep-learning-project-2025-dl_team_17/blob/main/results/models/flickr8k/baseline/checkpoints/training_curves.png)
*Figure 1: Training and validation loss curves for the baseline CNN-LSTM model.*

![Attention Training Curves](https://github.com/ML-DL-Teaching/deep-learning-project-2025-dl_team_17/blob/main/results/models/flickr8k/attention/checkpoints/training_curves.png)
*Figure 2: Training and validation loss curves for the attention-enhanced model.*

### 4.2 Training Performance

| Metric              | Baseline Model | Attention Model |
|---------------------|----------------|-----------------|
| Training Time       | `73.0 minutes`   | `76.6 minutes`    |
| Time per Epoch      | `7.3 minutes`    | `7.7 minutes`     |
| Initial Loss        | `4.891`          | `4.538`           |
| Final Loss          | `3.567`          | `2.912`           |
| Loss Reduction      | `27.0%`          | `35.8%`           |
| Trainable Params    | `4.03M`          | `5.66M`           |

### 4.3 Sample Results

#### Example: Normal Scene

| Model | Generated Caption |
|-------|------------------|
| **Ground Truth** | the dog is playinh in the water . |
| **Baseline** | a dog a in water |

![Baseline Caption Sample](https://github.com/ML-DL-Teaching/deep-learning-project-2025-dl_team_17/blob/main/results/models/flickr8k/baseline/checkpoints/visualizations/caption_sample_4.png)
*Figure 3: Example caption generation from the baseline model.*

| Model | Generated Caption |
|-------|------------------|
| **Ground Truth** | a biker does a tric on a ramp . |
| **Attention** | a man in a blue shirt is riding a bycicle on a bycicle |

![Attention Caption Sample](https://github.com/ML-DL-Teaching/deep-learning-project-2025-dl_team_17/blob/main/results/models/flickr8k/attention/checkpoints/visualizations/caption_sample_5.png)
*Figure 4: Example caption generation from the attention-enhanced model.*

![Attention Visualization](https://github.com/ML-DL-Teaching/deep-learning-project-2025-dl_team_17/blob/main/results/models/flickr8k/attention/checkpoints/visualizations/attention_sample_5.png)
*Figure 5: Visualization of attention weights for each generated word, highlighting which image regions the model focuses on during caption generation.*

### 4.4 Qualitative Analysis

The attention mechanism demonstrates superior performance in several key areas:

* Capturing fine-grained details and spatial relationships
* Correctly identifying multiple objects and their interactions
* Maintaining grammatical coherence in longer captions
* Providing interpretable decision-making through attention visualizations

As evidenced by the BLEU scores, the attention model achieves substantial improvements across all metrics. The most dramatic enhancement is observed in higher-order n-gram matching (`BLEU-3` and `BLEU-4`), indicating the model's superior ability to generate structurally coherent and contextually relevant captions. The attention mechanism enables the model to focus on relevant image regions when generating each word, resulting in more detailed and accurate descriptions.

### 4.5 Computational Considerations

| Resource Usage      | Baseline Model | Attention Model |
|---------------------|----------------|-----------------|
| GPU Memory (Train)  | `4.2 GB`         | `5.1 GB`          |
| GPU Memory (Infer)  | `1.8 GB`         | `2.3 GB`          |
| Inference Time      | `45 ms/image`    | `58 ms/image`     |
| Throughput          | `22 img/s`       | `17 img/s`        |

While the attention model requires approximately `6%` more parameters (`29.17M` vs. `27.54M`) and `5%` additional training time, the performance improvements justify the computational overhead for applications prioritizing caption quality. The attention model demonstrates faster convergence with a `35.8%` reduction in training loss compared to the baseline's `27.0%` reduction over the same number of epochs.

## 5. Documentation Structure

This repository includes comprehensive technical documentation:

1. **[Installation Guide](INSTALL.md)** - Environment configuration and setup procedures
2. **[Architecture Specifications](src/models/ARCHITECTURE.md)** - Complete neural network designs

## 6. Conclusions

This implementation demonstrates that attention mechanisms significantly enhance image captioning performance by enabling dynamic visual grounding during text generation. Key findings include:

* **Performance Gains**: The attention model shows dramatic improvements, particularly in higher-order BLEU scores
* **Interpretability**: Attention weights provide valuable insights into model behavior
* **Efficiency**: Minimal computational overhead (`5%` training time increase) for substantial quality improvements
* **Generalization**: Better handling of complex scenes with multiple objects and relationships

## 7. Future Work

Several promising directions for extending this research include:

* Implementation of transformer-based architectures
* Exploration of self-attention mechanisms
* Scaling to larger datasets (`COCO`, `Conceptual Captions`)
* Integration of object detection for enhanced spatial reasoning
* Multi-lingual caption generation

## References

1. Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., Zemel, R., & Bengio, Y. (2015). Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. *ICML*.
2. Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). Show and Tell: A Neural Image Caption Generator. *CVPR*.