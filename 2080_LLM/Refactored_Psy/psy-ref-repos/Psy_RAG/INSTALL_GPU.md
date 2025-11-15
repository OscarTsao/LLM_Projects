# GPU Installation Instructions for RTX 3090

This document provides instructions for installing the project with full GPU support for RTX 3090 optimization.

## Prerequisites

- NVIDIA RTX 3090 GPU
- CUDA 12.0+ (recommended)
- Python 3.8-3.11 (3.12+ may have compatibility issues with some PyTorch builds)
- 24GB+ RAM
- Ubuntu 20.04+ or compatible Linux distribution

## Installation Steps

### 1. CUDA Installation

Install CUDA Toolkit 12.0+:

```bash
# Download and install CUDA from NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. PyTorch with CUDA Support

Install PyTorch with CUDA support:

```bash
# For CUDA 12.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 3. FAISS GPU Support

Install FAISS with GPU support:

```bash
# Using conda (recommended)
conda install -c pytorch -c nvidia faiss-gpu

# Or using pip (may require specific versions)
pip install faiss-gpu
```

### 4. Project Dependencies

Install project dependencies:

```bash
# Install all dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### 5. RTX 3090 Specific Optimizations

The project includes automatic RTX 3090 optimizations:

- **Mixed Precision (FP16)**: Reduces memory usage by ~50%
- **TensorFloat-32 (TF32)**: Accelerates matrix operations
- **cuDNN Benchmark**: Optimizes convolution algorithms
- **Memory Management**: Uses 85% of GPU memory efficiently
- **Torch Compilation**: JIT compilation for faster inference

### 6. Verify GPU Setup

Run the performance test:

```bash
python test_performance.py
```

Expected output should show:
- CUDA available: True
- GPU detected: RTX 3090
- FAISS GPU support available

### 7. Alternative: CPU-Only Installation

If GPU setup fails, the project will automatically fallback to CPU:

```bash
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install CPU-only FAISS
pip install faiss-cpu
```

## Performance Expectations

### RTX 3090 Performance:
- **Embedding Generation**: ~1000-2000 posts/sec
- **FAISS Search**: <10ms for 10k criteria
- **End-to-end Processing**: ~100-200 posts/sec

### CPU Performance:
- **Embedding Generation**: ~50-100 posts/sec
- **FAISS Search**: <100ms for 10k criteria
- **End-to-end Processing**: ~10-20 posts/sec

## Troubleshooting

### Common Issues:

1. **CUDA Version Mismatch**:
   ```bash
   # Check CUDA version
   nvcc --version
   nvidia-smi
   
   # Install matching PyTorch version
   ```

2. **Out of Memory Errors**:
   - Reduce batch size in `src/config/settings.py`
   - Lower memory fraction in performance optimizer

3. **Slow Performance**:
   - Verify GPU utilization with `nvidia-smi`
   - Check that models are using GPU: `model.device`

4. **Import Errors**:
   - Verify all dependencies are installed
   - Check Python version compatibility
   - Try installing in a fresh virtual environment

## Memory Usage

RTX 3090 (24GB VRAM) allocation:
- BGE-M3 Model: ~3GB
- SpanBERT Model: ~2GB
- FAISS Index: ~1-5GB (depending on dataset size)
- Working Memory: ~10-15GB
- Reserved for System: ~2-3GB

Total: ~18-25GB (may require memory optimization for very large datasets)