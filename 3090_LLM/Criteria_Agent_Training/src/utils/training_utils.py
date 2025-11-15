# Standard library imports
import os

# Third-party imports
import torch


def get_optimized_training_params():
    """Get optimized training parameters based on available hardware"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if 'RTX 3090' in gpu_name:
            return {
                'batch_size': 128,
                'max_features': 8000,
                'hidden_dim': 512,
                'num_epochs': 20
            }
        else:
            return {
                'batch_size': 64,
                'max_features': 5000,
                'hidden_dim': 256,
                'num_epochs': 15
            }
    return {
        'batch_size': 32,
        'max_features': 3000,
        'hidden_dim': 128,
        'num_epochs': 10
    }


def validate_data_files(*file_paths):
    """Validate that all required data files exist"""
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required data file not found: {file_path}")


def print_gpu_info():
    """Print GPU information and optimization settings"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        params = get_optimized_training_params()
        print(f"Optimized parameters: {params}")
    else:
        print("No GPU available, using CPU")