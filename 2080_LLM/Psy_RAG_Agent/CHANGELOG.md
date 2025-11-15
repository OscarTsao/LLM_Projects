# Changelog

## [Enhanced Metrics and Hydra Configuration] - 2025-09-15

### Added
- **Enhanced Evaluation Metrics**: Added comprehensive metrics including:
  - Individual precision and recall for each class
  - Overall precision and recall averages
  - Weighted precision, recall, and F1 scores
  - Confusion matrix visualization
  - Per-class detailed metrics

- **Hydra Configuration Management**: Implemented Hydra for parameter management
  - Modular configuration system with separate files for training and model parameters
  - Multiple preset configurations:
    - `training=default`: Balanced training parameters
    - `training=fast`: Quick training for testing (5 epochs)
    - `training=thorough`: Comprehensive training for best results (30+ epochs)
    - `model=basic`: Standard model configuration
    - `model=small`: Lightweight model for limited resources
    - `model=large`: High-capacity model for maximum performance
  - GPU auto-optimization for RTX 3090 and other GPUs
  - Easy command-line parameter overrides

### Enhanced
- **Training Pipeline**: Updated BasicTrainer to provide more detailed evaluation output
- **Metrics Calculation**: Replaced simple metrics with comprehensive evaluation suite
- **Configuration Flexibility**: Can now easily adjust parameters without modifying code

### New Files
- `hydra_trainer.py`: Main training script with Hydra support
- `conf/config.yaml`: Default Hydra configuration
- `conf/training/`: Training configuration presets (default, fast, thorough)
- `conf/model/`: Model configuration presets (basic, small, large)
- `test_enhanced_metrics.py`: Test script for metrics functionality
- `test_hydra_config.py`: Test script for Hydra configuration

### Usage Examples

**Basic training with Hydra:**
```bash
python hydra_trainer.py
```

**Fast training with small model:**
```bash
python hydra_trainer.py training=fast model=small
```

**Thorough training with large model:**
```bash
python hydra_trainer.py training=thorough model=large
```

**Override specific parameters:**
```bash
python hydra_trainer.py training.num_epochs=25 model.max_features=10000
```

### Dependencies Added
- `hydra-core>=1.3.0`
- `omegaconf>=2.3.0`