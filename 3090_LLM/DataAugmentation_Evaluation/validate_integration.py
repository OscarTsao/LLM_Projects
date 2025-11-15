#!/usr/bin/env python3
"""Integration validation script for the multi-agent system."""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check that all required dependencies are available."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers',
        'hydra-core',
        'optuna',
        'mlflow',
        'sklearn',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package}")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        return False
    
    logger.info("All dependencies available!")
    return True


def check_data_files():
    """Check that required data files exist."""
    logger.info("Checking data files...")
    
    required_files = [
        "Data/GroundTruth/Final_Ground_Truth.json",
        "Data/ReDSM5/redsm5_posts.csv",
        "Data/ReDSM5/redsm5_annotations.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"✓ {file_path}")
        else:
            missing_files.append(file_path)
            logger.error(f"✗ {file_path}")
    
    if missing_files:
        logger.error(f"Missing data files: {missing_files}")
        return False
    
    logger.info("All data files present!")
    return True


def check_configuration_files():
    """Check that configuration files are valid."""
    logger.info("Checking configuration files...")
    
    config_files = [
        "conf/agent/criteria.yaml",
        "conf/agent/evidence.yaml", 
        "conf/agent/joint.yaml",
        "conf/training_mode/criteria.yaml",
        "conf/training_mode/evidence.yaml",
        "conf/training_mode/joint.yaml"
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ {config_file}")
        except Exception as e:
            logger.error(f"✗ {config_file}: {e}")
            return False
    
    logger.info("All configuration files valid!")
    return True


def test_agent_imports():
    """Test that all agent modules can be imported."""
    logger.info("Testing agent imports...")
    
    try:
        from src.agents.base import BaseAgent, AgentOutput, CriteriaMatchingConfig, EvidenceBindingConfig
        from src.agents.criteria_matching import CriteriaMatchingAgent
        from src.agents.evidence_binding import EvidenceBindingAgent
        from src.agents.multi_agent_pipeline import MultiAgentPipeline, JointTrainingModel
        logger.info("✓ All agent modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Agent import failed: {e}")
        return False


def test_data_loading():
    """Test that data loading modules work."""
    logger.info("Testing data loading...")
    
    try:
        from src.data.evidence_loader import load_evidence_annotations
        from src.data.joint_dataset import create_joint_dataset
        from src.data.redsm5_loader import load_ground_truth_frame
        
        # Test loading a small sample
        if Path("Data/ReDSM5/redsm5_posts.csv").exists():
            examples = load_evidence_annotations(
                "Data/ReDSM5/redsm5_posts.csv",
                "Data/ReDSM5/redsm5_annotations.csv"
            )
            logger.info(f"✓ Loaded {len(examples)} evidence examples")
        
        if Path("Data/GroundTruth/Final_Ground_Truth.json").exists():
            ground_truth = load_ground_truth_frame("Data/GroundTruth/Final_Ground_Truth.json")
            logger.info(f"✓ Loaded {len(ground_truth)} ground truth examples")
        
        return True
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return False


def test_agent_creation():
    """Test that agents can be created and run inference."""
    logger.info("Testing agent creation and inference...")
    
    try:
        from src.agents.base import CriteriaMatchingConfig, EvidenceBindingConfig
        from src.agents.criteria_matching import CriteriaMatchingAgent
        from src.agents.evidence_binding import EvidenceBindingAgent
        from src.agents.multi_agent_pipeline import MultiAgentPipeline
        
        # Create test configurations
        criteria_config = CriteriaMatchingConfig(
            model_name="google-bert/bert-base-uncased",
            max_seq_length=128,
            dropout=0.1,
            classifier_hidden_sizes=[64]
        )
        
        evidence_config = EvidenceBindingConfig(
            model_name="google-bert/bert-base-uncased",
            max_seq_length=128,
            dropout=0.1
        )
        
        # Create agents
        criteria_agent = CriteriaMatchingAgent(criteria_config)
        evidence_agent = EvidenceBindingAgent(evidence_config)
        pipeline = MultiAgentPipeline(criteria_agent, evidence_agent)
        
        # Test inference
        posts = ["I feel very sad and depressed."]
        criteria = ["Depressed mood most of the day."]
        
        outputs = pipeline.predict_batch(posts, criteria)
        
        logger.info("✓ Agent creation and inference successful")
        logger.info(f"  Criteria match: {outputs.criteria_match.item():.3f}")
        logger.info(f"  Criteria confidence: {outputs.criteria_confidence.item():.3f}")
        logger.info(f"  Evidence spans: {len(outputs.evidence_spans[0])}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Agent creation/inference failed: {e}")
        return False


def test_training_scripts():
    """Test that training scripts can be imported and configured."""
    logger.info("Testing training script imports...")
    
    try:
        from src.training.train_criteria import create_criteria_agent
        from src.training.train_evidence import create_evidence_agent
        from src.training.train_joint import create_joint_model
        
        # Test Optuna scripts
        from src.training.train_criteria_optuna import suggest_hyperparameters
        from src.training.train_evidence_optuna import suggest_hyperparameters as suggest_evidence_hyperparameters
        
        logger.info("✓ All training scripts imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Training script import failed: {e}")
        return False


def test_configuration_loading():
    """Test loading configurations with Hydra."""
    logger.info("Testing configuration loading...")
    
    try:
        # Test loading different training mode configs
        from hydra import initialize, compose
        from hydra.core.global_hydra import GlobalHydra
        
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        with initialize(version_base=None, config_path="conf"):
            # Test criteria config
            cfg = compose(config_name="config", overrides=["training_mode=criteria"])
            logger.info("✓ Criteria training mode config loaded")
            
            # Test evidence config  
            cfg = compose(config_name="config", overrides=["training_mode=evidence"])
            logger.info("✓ Evidence training mode config loaded")
            
            # Test joint config
            cfg = compose(config_name="config", overrides=["training_mode=joint"])
            logger.info("✓ Joint training mode config loaded")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        return False


def test_hardware_setup():
    """Test hardware setup and optimizations."""
    logger.info("Testing hardware setup...")
    
    try:
        from src.agents.base import setup_hardware_optimizations, get_device_info
        
        # Setup optimizations
        setup_hardware_optimizations()
        
        # Get device info
        device_info = get_device_info()
        logger.info(f"✓ CUDA available: {device_info['cuda_available']}")
        logger.info(f"✓ Device count: {device_info['device_count']}")
        
        if device_info['cuda_available']:
            logger.info(f"✓ Device name: {device_info['device_name']}")
            logger.info(f"✓ Memory allocated: {device_info['memory_allocated'] / 1e9:.2f} GB")
        
        return True
    except Exception as e:
        logger.error(f"✗ Hardware setup failed: {e}")
        return False


def run_integration_tests():
    """Run all integration tests."""
    logger.info("Starting integration validation...")
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Data Files", check_data_files),
        ("Configuration Files", check_configuration_files),
        ("Agent Imports", test_agent_imports),
        ("Data Loading", test_data_loading),
        ("Agent Creation", test_agent_creation),
        ("Training Scripts", test_training_scripts),
        ("Configuration Loading", test_configuration_loading),
        ("Hardware Setup", test_hardware_setup),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All integration tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed!")
        return False


def main():
    """Main function."""
    success = run_integration_tests()
    
    if success:
        logger.info("\n✅ Integration validation completed successfully!")
        logger.info("The multi-agent system is ready for use.")
        logger.info("\nNext steps:")
        logger.info("1. Run 'make train-criteria' to test criteria matching training")
        logger.info("2. Run 'make train-evidence' to test evidence binding training")
        logger.info("3. Run 'make train-joint' to test joint training")
        logger.info("4. Run 'make mlflow-ui' to monitor experiments")
        sys.exit(0)
    else:
        logger.error("\n❌ Integration validation failed!")
        logger.error("Please fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
