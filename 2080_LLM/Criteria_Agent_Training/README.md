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
pip install pandas numpy
```

### 2. Test the Pipeline

```bash
# Quick test with small dataset
python quick_test.py

# Test data loading only
python test_data_loading.py
```

### 3. Full Training

```bash
# Run optimized training for RTX 3090
python run_full_training.py

# Or use the basic classifier directly
python basic_classifier.py
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
├── basic_classifier.py               # Main classifier (recommended)
├── run_full_training.py             # Optimized full training script
├── quick_test.py                     # Quick test script
├── test_data_loading.py              # Data loading test
│
├── spanbert_classifier.py            # Advanced SpanBERT implementation
├── spanbert_simple.py               # Simplified SpanBERT
├── minimal_classifier.py            # Minimal implementation
│
├── train_and_evaluate.py            # Generic training script
├── demo_prediction.py              # Prediction demo
│
└── Data/
    ├── translated_posts.csv         # Input posts
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
from basic_classifier import BasicTrainer

# Load trained model
trainer = BasicTrainer()
trainer.model = load_trained_model()

# Make prediction
post = "I've been feeling very sad and hopeless lately..."
criterion = "Depressed mood most of the day, nearly every day"

result = predict_match(post, criterion)
print(f"Match probability: {result['probability']}")
```

### Custom Training

```python
from basic_classifier import BasicTrainer

# Initialize with custom parameters
trainer = BasicTrainer(max_features=10000)

# Load your data
df = trainer.load_and_prepare_data(posts_path, criteria_path, labels_path)

# Train
train_ds, val_ds, test_ds = trainer.create_datasets(df)
trainer.train(train_ds, val_ds, num_epochs=25, batch_size=256)
```

## Advanced Features

### Model Variants

1. **`basic_classifier.py`** (Recommended): Fast, reliable, GPU-optimized
2. **`spanbert_classifier.py`**: Advanced transformer-based (requires compatible transformers)
3. **`minimal_classifier.py`**: Lightweight version for limited resources

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

- Use the `basic_classifier.py` for most reliable results
- Increase `max_features` for better vocabulary coverage
- Use larger `batch_size` if you have sufficient GPU memory
- Monitor GPU utilization with `nvidia-smi`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test with `python quick_test.py`
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