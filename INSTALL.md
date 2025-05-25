# Installation Guide

## Abstract
This document provides detailed installation procedures for the image captioning project, covering environment setup, dependency management, and dataset acquisition. 

Whether you're using GPU acceleration or CPU-only computation, this guide ensures a smooth setup process for reproducing our experimental results.

## 1. System Requirements

### 1.1 Hardware Requirements

| Component | Minimum | Recommended | Our Setup |
|-----------|---------|-------------|-----------|
| GPU | `4GB VRAM` | `6GB+ VRAM` | `8GB VRAM` |
| RAM | `8GB` | `16GB` | `16GB` |
| Storage | `10GB` | `20GB SSD` | `50GB SSD` |
| CPU | `4 cores` | `6+ cores` | `8 cores` |

### 1.2 Software Prerequisites
- Operating System: Linux (Ubuntu `18.04+`), macOS (`10.14+`), or Windows `10/11`
- CUDA: `11.8` or higher (for GPU support)
  - We are using CUDA `11.8`
- Python: `3.8` or higher
  - We are using `3.10`
- Conda: Anaconda or Miniconda
  - Better Miniconda using mamba (substitue `conda` per `mamba`)
- Git: For repository cloning

## 2. Environment Setup

### 2.1 Clone Repository
```bash
# Clone the repository
git clone https://github.com/ML-DL-Teaching/deep-learning-project-2025-dl_team_17.git
cd deep-learning-project-2025-dl_team_17

# Verify repository structure
ls -la
```

### 2.2 Create Conda Environment
The project includes an `environment.yml` file for easy setup:

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate image-captioning

# Check Python version
python --version
```

### 2.3 Alternative: Manual Environment Setup
If you prefer manual installation or encounter issues with the YAML file:

```bash
# Create new environment
conda create -n image-captioning python=3.10

# Activate environment
conda activate image-captioning

# Install PyTorch with CUDA support
conda install pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional dependencies
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 3. Dataset Setup

### 3.1 Flickr8k Dataset Structure
The `Flickr8k` dataset must be organized as follows:

```
data/
└── raw/
    ├── images/           # 8,000 JPEG images
    │   ├── 1000268201_693b08cb0e.jpg
    │   ├── 1001773457_577c3a7d70.jpg
    │   └── ...
    └── captions.txt      # Image-caption pairs
```

### 3.2 Dataset Download

#### Option 1: Official Source
1. Request access from [University of Illinois](https://forms.illinois.edu/sec/1713398)
2. Download the dataset archive (approximately `1GB`)
3. Extract to `data/raw/` directory

#### Option 2: Kaggle (Alternative)
```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset (requires Kaggle account)
kaggle datasets download -d adityajn105/flickr8k -p data/

# Extract files
cd data/
unzip flickr8k.zip -d raw/
cd ..

# Verify dataset structure
ls -la data/raw/
```

## 4. NLTK Data Setup
Download required NLTK resources for text preprocessing:

```python
# Run in Python interpreter or script
import nltk

# Download required data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')  # Additional resource for preprocessing

print("NLTK data downloaded successfully!")
```

## 5. Weights & Biases Configuration
Set up experiment tracking for monitoring training progress:

```bash
# Install W&B if not already installed
pip install wandb

# Login to W&B (optional but recommended)
wandb login

# Or set offline mode for local logging (Very useful as it can be synchronized afterwards)
export WANDB_MODE=offline

# Verify W&B installation
python -c "import wandb; print(f'W&B version: {wandb.__version__}')"
```

## 6. Configuration Files
Verify all configuration files are present:

```bash
# Check config directory
ls config/

# View configuration contents
cat config/data.yaml
```

Expected files:
- `data.yaml` - Dataset configuration
- `model.yaml` - Model hyperparameters  
- `training.yaml` - Training settings
- `wandb.yaml` - Experiment tracking

## 7. Verification Script
Run the verification script to ensure proper installation:

```bash
# Test basic imports and GPU availability
python -c "
import torch
import torchvision
import nltk
import numpy as np
import pandas as pd
from PIL import Image

print('All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"
```

## 8. Troubleshooting

### 8.1 Common Issues

| Issue | Solution |
|-------|----------|
| CUDA not found | Verify GPU drivers with `nvidia-smi` |
| Out of memory | Reduce `batch_size` in `config/training.yaml` |
| Dataset not found | Check paths in `config/data.yaml` |
| Package conflicts | Create fresh conda environment |
| Permission denied | Use `chmod +x` for scripts |

### 8.2 GPU Memory Settings
For limited GPU memory:

```yaml
# config/training.yaml
training:
  batch_size: 64  # Reduce from 128
  gradient_accumulation_steps: 4
  mixed_precision: true  # Enable for memory efficiency
```

### 8.3 CPU-Only Setup
For CPU-only training (significantly slower):

```yaml
# config/training.yaml
device: cpu
training:
  batch_size: 16
  num_workers: 0
  pin_memory: false
```

## 9. Next Steps
After successful installation:

1. **Train baseline model**: `python scripts/baseline.py`
2. **Train attention model**: `python scripts/attention.py`
3. **Analyze results**: Open notebooks in `notebooks/` directory
4. **Visualize attention**: Use tools in `src/visualization/`
5. **Compare models**: Run `python src/comparison/compare_models.py`

## 10. Environment Management

### 10.1 Export Environment
Save exact environment for reproducibility:

```bash
# Export with conda
conda env export > environment.yml

# Export pip requirements
pip freeze > requirements.txt

# Alternative (minimal environment file)
conda env export --from-history > environment.yml
```

### 10.2 Update Environment
To update packages:

```bash
# Update conda packages
conda update --all

# Update specific package
conda update pytorch torchvision

# Update pip packages
pip install --upgrade -r requirements.txt
```

### 10.3 Remove Environment
To clean up:

```bash
# Deactivate environment
conda deactivate

# Remove environment
conda env remove -n image-captioning

# Clean conda cache
conda clean --all
```

## 11. Additional Resources
- **Project Repository**: [GitHub](https://github.com/ML-DL-Teaching/deep-learning-project-2025-dl_team_17)
- **PyTorch Documentation**: [pytorch.org](https://pytorch.org/docs/stable/index.html)
- **CUDA Installation**: [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
- **Conda Cheatsheet**: [Conda Docs](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)