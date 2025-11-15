# Criteria Evidence Agent - Jupyter Notebook System

## 🚀 Overview

This repository contains a completely refactored version of the Criteria Evidence Agent project, transformed from Python scripts into an interactive Jupyter notebook system with enhanced auto-resume capabilities, comprehensive checkpointing, and advanced hyperparameter optimization.

## ✨ Key Features

### 🔧 Enhanced Configuration Management
- **Interactive Configuration Builder**: Create configurations using intuitive widgets
- **Configuration Validation**: Automatic validation with detailed error reporting
- **Preset Configurations**: Pre-defined configurations for common use cases
- **Configuration Comparison**: Compare different configurations side-by-side
- **Dynamic Configuration**: Real-time configuration updates and previews

### 💾 Advanced Checkpoint System
- **Complete State Preservation**: Save model, optimizer, scheduler, and training metadata
- **Auto-Resume Training**: Automatically detect and resume interrupted training
- **Configuration Compatibility**: Verify configuration compatibility before resuming
- **Random State Management**: Preserve random states for reproducibility
- **Intelligent Cleanup**: Automatic cleanup of old checkpoints

### 🏃 Enhanced Training System
- **Interactive Training**: Start/stop training with interactive widgets
- **Real-time Monitoring**: Live progress bars and metric visualization
- **MLflow Integration**: Comprehensive experiment tracking
- **EMA Support**: Exponential Moving Average for model weights
- **Mixed Precision**: Automatic mixed precision training support

### 🔍 Advanced HPO System
- **Optuna Integration**: State-of-the-art hyperparameter optimization
- **Auto-Resume HPO**: Resume interrupted optimization studies
- **Comprehensive Search Space**: Extensive hyperparameter search definitions
- **Trial Pruning**: Intelligent early stopping for efficiency
- **Results Visualization**: Interactive plots and analysis dashboards

### 📊 Monitoring and Visualization
- **Interactive Dashboards**: Real-time training and HPO monitoring
- **Progress Visualization**: Training history and metric plots
- **Experiment Comparison**: Compare multiple experiments and studies
- **Export Capabilities**: Export best configurations and results

## 📁 Notebook Structure

```
├── 01_Configuration_Management.ipynb       # Configuration system and management
├── 02_Enhanced_Checkpoint_System.ipynb     # Checkpoint and auto-resume system
├── 03_Main_Training.ipynb                  # Enhanced training with auto-resume
├── 04_HPO_Optimization.ipynb               # Hyperparameter optimization
├── 05_Code_Verification.ipynb              # System verification and testing
├── 06_Data_Processing_and_Exploration.ipynb # Data analysis and preprocessing
├── 07_Model_Evaluation.ipynb               # Model evaluation and analysis
└── README_NOTEBOOK_SYSTEM.md               # This documentation
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install torch transformers optuna mlflow pandas numpy scikit-learn
pip install matplotlib seaborn plotly tqdm ipywidgets jupyter
pip install omegaconf peft accelerate

# Start Jupyter
jupyter notebook
```

### 2. Configuration Setup

1. Open `01_Configuration_Management.ipynb`
2. Run all cells to initialize the configuration system
3. Use the interactive configuration builder to create your setup
4. Save your configuration with a meaningful name

### 3. Data Processing and Exploration

1. Open `06_Data_Processing_and_Exploration.ipynb`
2. Configure data paths and parameters
3. Load and explore your dataset
4. Analyze text statistics and label distributions
5. Assess data quality and get recommendations

### 4. Training

1. Open `03_Main_Training.ipynb`
2. Run the setup cells
3. Select your configuration using the dropdown
4. Click "Start Training" to begin
5. Monitor progress in real-time

### 5. Hyperparameter Optimization

1. Open `04_HPO_Optimization.ipynb`
2. Select a base configuration
3. Configure HPO parameters (trials, jobs, etc.)
4. Click "Start HPO" to begin optimization
5. Analyze results using the dashboard

### 6. Model Evaluation

1. Open `07_Model_Evaluation.ipynb`
2. Select a trained model for evaluation
3. Run comprehensive evaluation on test/validation data
4. Analyze performance metrics and error patterns
5. Compare multiple models and export results

### 7. Verification

1. Open `05_Code_Verification.ipynb`
2. Run all cells to verify system integrity
3. Check the verification report for any issues

## 🔄 Auto-Resume Capabilities

### Training Auto-Resume
- **Automatic Detection**: Detects interrupted training sessions
- **Configuration Validation**: Ensures configuration compatibility
- **State Restoration**: Restores complete training state
- **Progress Continuation**: Continues from exact interruption point

### HPO Auto-Resume
- **Study Persistence**: Saves Optuna study state automatically
- **Trial Continuation**: Resumes from last completed trial
- **Progress Tracking**: Maintains optimization history
- **Result Preservation**: Preserves all trial results and metadata

## 📊 Configuration System

### Configuration Classes
- `ExperimentConfig`: Main configuration container
- `DataConfig`: Data loading and preprocessing settings
- `ModelConfig`: Model architecture and parameters
- `TrainingConfig`: Training hyperparameters and settings
- `HPOConfig`: Hyperparameter optimization settings

### Interactive Features
- **Widget-based Builder**: Create configurations using interactive widgets
- **Real-time Validation**: Immediate feedback on configuration issues
- **Preset Management**: Save and load configuration presets
- **Comparison Tools**: Compare configurations side-by-side

## 💾 Checkpoint System

### Checkpoint Contents
- **Model State**: Complete model weights and architecture
- **Optimizer State**: Optimizer parameters and momentum
- **Scheduler State**: Learning rate scheduler state
- **Training Metadata**: Epoch, step, metrics, and progress
- **Random States**: PyTorch, NumPy, and CUDA random states
- **Configuration**: Complete experiment configuration

### Auto-Resume Logic
1. **Detection**: Check for existing checkpoints for the experiment
2. **Validation**: Verify configuration compatibility
3. **Loading**: Restore complete training state
4. **Continuation**: Resume training from exact point

## 🔍 HPO System

### Search Space
- **Model Architecture**: Encoder types, pooling strategies
- **Training Parameters**: Learning rates, batch sizes, optimizers
- **Regularization**: Dropout rates, weight decay, label smoothing
- **Advanced Features**: LoRA parameters, EMA decay, focal loss

### Optimization Features
- **TPE Sampler**: Tree-structured Parzen Estimator
- **Median Pruner**: Early stopping for unpromising trials
- **Parallel Execution**: Multi-job optimization support
- **Progress Tracking**: Real-time optimization monitoring

## 📈 Monitoring and Visualization

### Training Monitoring
- **Real-time Plots**: Loss curves and metric progression
- **Progress Bars**: Detailed training progress with ETA
- **Metric Tracking**: Comprehensive metric logging
- **Checkpoint Status**: Visual checkpoint management

### HPO Analysis
- **Optimization History**: Trial progression visualization
- **Parameter Importance**: Feature importance analysis
- **Best Trial Analysis**: Detailed best configuration analysis
- **Study Comparison**: Compare multiple optimization studies

## 🛠️ System Requirements

### Dependencies
- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- Optuna 3.0+
- MLflow 2.0+
- Jupyter Notebook/Lab

### Hardware
- **GPU**: CUDA-compatible GPU recommended
- **Memory**: 16GB+ RAM recommended
- **Storage**: SSD recommended for checkpoint I/O

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   - Run `05_Code_Verification.ipynb` to check dependencies
   - Install missing packages with pip/conda

2. **Configuration Issues**
   - Use configuration validation in the management notebook
   - Check data paths and file existence

3. **Checkpoint Problems**
   - Verify checkpoint directory permissions
   - Check available disk space

4. **Training Interruptions**
   - Enable auto-resume in configuration
   - Verify checkpoint saving frequency

### Verification
Run the verification notebook to check:
- All imports and dependencies
- Configuration system functionality
- Checkpoint system integrity
- Model and data loading
- Training component compatibility
- HPO system functionality

## 📚 Advanced Usage

### Custom Configurations
```python
# Create custom configuration
config = ExperimentConfig()
config.model.encoder.type = "deberta"
config.training.batch_size = 32
config.training.optimizer.learning_rate = 1e-5

# Save for reuse
config_manager.save_config(config, "my_custom_config")
```

### Manual Checkpoint Management
```python
# List checkpoints
checkpoints = checkpoint_manager.list_checkpoints("my_experiment")

# Load specific checkpoint
training_state, config = checkpoint_manager.load_checkpoint(
    checkpoint_id, model, optimizer, scheduler, device
)

# Delete old checkpoints
checkpoint_manager.delete_checkpoint(checkpoint_id)
```

### HPO Customization
```python
# Custom search space
def custom_suggest_hyperparameters(trial, base_config):
    config = base_config.copy()
    config.training.learning_rate = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    config.model.encoder.dropout = trial.suggest_float("dropout", 0.0, 0.5)
    return config

# Custom objective function
def custom_objective(trial):
    config = custom_suggest_hyperparameters(trial, base_config)
    result = train_model(config)
    return result["best_metric"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run verification tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original Criteria Evidence Agent project team
- Optuna team for excellent HPO framework
- MLflow team for experiment tracking
- Jupyter team for interactive computing platform
