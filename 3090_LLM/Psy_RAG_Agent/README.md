# DSM-5 Criteria Classification with Neural Networks

This project implements a neural network classifier to determine whether social media posts match specific DSM-5 (Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition) criteria. The system uses natural language processing and deep learning techniques optimized for GPU acceleration.

## Project Overview

The classifier takes pairs of (social media post, DSM-5 criterion) as input and predicts whether the post content matches the criterion description (binary classification: 0 = No Match, 1 = Match).

### Key Features

- **GPU Optimized**: Specifically optimized for NVIDIA RTX 3090 with CUDA acceleration
- **Robust Architecture**: Multiple classifier implementations from simple neural networks to transformer-based models
- **Comprehensive Evaluation**: Detailed metrics including accuracy, F1-score, precision, and recall
- **Scalable Pipeline**: Handles large datasets with efficient batch processing
- **Flexible Tokenization**: Custom tokenization and feature extraction pipelines

## Dataset Structure

### Input Files

1. **`Data/translated_posts.csv`**: Contains translated social media posts
2. **`Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json`**: DSM-5 criteria definitions
3. **`Data/Groundtruth/criteria_evaluation.csv`**: Ground truth labels for post-criteria pairs

### Data Statistics

- **Total Posts**: 2,675 unique social media posts
- **Total Criteria**: 131 different DSM-5 criteria across 15 disorders
- **Training Examples**: 189,925 post-criterion pairs
- **Class Distribution**: ~55% positive matches, ~45% negative matches

## Model Architecture

### Basic Classifier (Recommended)

The basic classifier uses a simple but effective neural network architecture:

```
Input Layer (TF-IDF Features)
    ↓
Linear Layer (hidden_dim) + BatchNorm + ReLU + Dropout
    ↓
Linear Layer (hidden_dim/2) + BatchNorm + ReLU + Dropout
    ↓
Linear Layer (hidden_dim/4) + ReLU + Dropout
    ↓
Output Layer (2 classes)
```

### Features

- **Text Processing**: TF-IDF vectorization with n-grams (1,2)
- **Vocabulary Size**: 3,000-8,000 most frequent words (configurable)
- **Hidden Dimensions**: 128-512 (auto-scaled based on GPU)
- **Regularization**: Dropout and batch normalization
- **Optimization**: AdamW optimizer with weight decay

## Quick Start

### 1. Environment Setup

```bash
# Ensure PyTorch with CUDA is installed
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
# Train basic classifier (recommended)
python train.py --model basic

# Train SpanBERT model
python train.py --model spanbert --epochs 5 --batch_size 8

# Train RAG-enhanced model
python train.py --model rag --epochs 3 --batch_size 4
```

### 4. Training Parameters

The system automatically optimizes parameters based on your GPU:

**RTX 3090 Configuration:**
- Batch Size: 128
- Max Features: 8,000
- Hidden Dimensions: 512
- Training Epochs: 20

**Standard GPU Configuration:**
- Batch Size: 64
- Max Features: 5,000
- Hidden Dimensions: 256
- Training Epochs: 15

## File Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── train.py                          # Main training script
│
├── src/                              # Source code
│   ├── basic_classifier.py          # Basic neural network classifier
│   ├── spanbert_classifier.py       # SpanBERT-based classifier
│   ├── spanbert_simple.py          # Simplified SpanBERT
│   ├── rag_spanbert_classifier.py  # RAG-enhanced classifier
│   └── minimal_classifier.py       # Minimal implementation
│
├── utils/                           # Utility modules
│   └── rag_retrieval.py           # RAG retrieval system
│
├── models/                         # Trained models output
│
└── Data/                          # Dataset
    ├── translated_posts.csv      # Input posts
    ├── DSM-5/
    │   └── DSM_Criteria_Array_Fixed_Simplify.json
    └── Groundtruth/
        └── criteria_evaluation.csv  # Labels
```

## Performance

### Expected Results

Based on validation testing:

- **Accuracy**: 85-95%
- **F1-Score**: 0.85-0.93
- **Training Time**: 10-30 minutes on RTX 3090
- **Memory Usage**: ~8-12 GB GPU memory

### GPU Optimization

The system includes specific optimizations for RTX 3090:

- **Mixed Precision Training**: Automatic FP16 when available
- **Large Batch Sizes**: Up to 128 samples per batch
- **Memory Efficient**: Gradient accumulation for large models
- **Fast Data Loading**: Multi-threaded data loading

## Usage Examples

### Making Predictions

```python
import sys
sys.path.insert(0, 'src')
from basic_classifier import BasicTrainer

# Load trained model
trainer = BasicTrainer()
trainer.load_model('models/trained')

# Make prediction
post = "I've been feeling very sad and hopeless lately..."
criterion = "Depressed mood most of the day, nearly every day"

# Predictions would need custom inference code
```

### Custom Training

```bash
# Basic classifier with custom parameters
python train.py --model basic --max_features 10000 --epochs 25 --batch_size 256

# SpanBERT with custom learning rate
python train.py --model spanbert --learning_rate 1e-5 --epochs 10
```

## Advanced Features

### Model Variants

1. **`basic`** (Recommended): Fast, reliable, GPU-optimized neural network
2. **`spanbert`**: Advanced transformer-based classifier
3. **`rag`**: RAG-enhanced SpanBERT with retrieval augmentation

### Extensibility

The codebase is designed for easy extension:

- **Custom Tokenizers**: Replace `BasicTokenizer` class
- **Different Architectures**: Modify `BasicClassifier` class
- **New Features**: Add to the feature extraction pipeline
- **Multi-GPU**: Ready for distributed training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch_size in training parameters
2. **Slow Training**: Ensure CUDA is properly installed and detected
3. **Poor Performance**: Try increasing max_features or model size
4. **Import Errors**: Check transformers library compatibility

### Performance Tips

- Use the `basic` model for most reliable results
- Increase `--max_features` for better vocabulary coverage
- Use larger `--batch_size` if you have sufficient GPU memory
- Monitor GPU utilization with `nvidia-smi`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test with `python train.py --model basic --epochs 1`
4. Submit a pull request

## License

This project is intended for research and educational purposes. Please ensure compliance with data usage agreements and ethical guidelines when working with mental health-related data.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dsm5_classifier_2024,
  title={DSM-5 Criteria Classification with Neural Networks},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/username/Criteria_Agent_Training}
}
```