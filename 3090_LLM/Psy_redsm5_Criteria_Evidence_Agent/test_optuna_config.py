#!/usr/bin/env python3
"""
Test script to verify Optuna configuration works correctly.
Tests database creation, study management, and parameter suggestion.
"""

import tempfile
from pathlib import Path
from omegaconf import OmegaConf
import optuna
import sys

def test_optuna_config():
    """Test the Optuna configuration for correctness."""
    print("🧪 Testing Optuna configuration...")
    
    # Load the maxed HPO config
    try:
        cfg = OmegaConf.load('configs/training/maxed_hpo.yaml')
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Test database creation with temporary file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        temp_storage = f"sqlite:///{tmp_db.name}"
        
        try:
            # Test study creation
            study = optuna.create_study(
                study_name="test_study",
                direction="maximize",
                storage=temp_storage,
                load_if_exists=False,
            )
            print("✓ Study creation successful")
            
            # Test parameter suggestion
            def test_objective(trial):
                # Test all parameter types from the search space
                params = {}
                
                # Test categorical parameters
                params['loss_function'] = trial.suggest_categorical(
                    'loss_function', 
                    cfg.search_space.loss_function.choices
                )
                params['optimizer_type'] = trial.suggest_categorical(
                    'optimizer_type',
                    cfg.search_space.optimizer_type.choices
                )
                
                # Test uniform parameters
                params['dropout'] = trial.suggest_float(
                    'dropout',
                    cfg.search_space.dropout.low,
                    cfg.search_space.dropout.high
                )
                params['alpha'] = trial.suggest_float(
                    'alpha',
                    cfg.search_space.alpha.low,
                    cfg.search_space.alpha.high
                )
                
                # Test loguniform parameters
                params['learning_rate'] = trial.suggest_float(
                    'learning_rate',
                    cfg.search_space.learning_rate.low,
                    cfg.search_space.learning_rate.high,
                    log=True
                )
                
                print(f"  📋 Sample parameters: {params}")
                return 0.85  # Dummy metric value
            
            # Run a few test trials
            study.optimize(test_objective, n_trials=3)
            print("✓ Parameter suggestion and trial execution successful")
            
            # Test study persistence
            study2 = optuna.load_study(study_name="test_study", storage=temp_storage)
            assert len(study2.trials) == 3, "Study persistence failed"
            print("✓ Study persistence successful")
            
            # Test best trial access
            best_trial = study2.best_trial
            print(f"✓ Best trial access successful (value: {best_trial.value})")
            
        except Exception as e:
            print(f"❌ Optuna test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Cleanup
            Path(tmp_db.name).unlink(missing_ok=True)
    
    print("🎉 All Optuna configuration tests passed!")
    return True

def test_search_space_coverage():
    """Test that the search space covers all important parameters."""
    print("\n🔍 Testing search space coverage...")
    
    cfg = OmegaConf.load('configs/training/maxed_hpo.yaml')
    search_space = cfg.search_space
    
    # Check for essential parameters
    essential_params = [
        'learning_rate', 'weight_decay', 'dropout', 'loss_function',
        'optimizer_type', 'scheduler_type', 'threshold'
    ]
    
    missing_params = []
    for param in essential_params:
        if param not in search_space:
            missing_params.append(param)
    
    if missing_params:
        print(f"❌ Missing essential parameters: {missing_params}")
        return False
    
    print("✓ All essential parameters present in search space")
    
    # Check parameter ranges are reasonable
    checks = [
        ('learning_rate', 'low', lambda x: x >= 1e-7),
        ('learning_rate', 'high', lambda x: x <= 1e-3),
        ('dropout', 'low', lambda x: x >= 0.0),
        ('dropout', 'high', lambda x: x <= 1.0),
        ('weight_decay', 'low', lambda x: x >= 1e-8),
    ]
    
    for param, bound, check_func in checks:
        if param in search_space:
            value = search_space[param][bound]
            if not check_func(value):
                print(f"❌ Parameter {param}.{bound} has unreasonable value: {value}")
                return False
    
    print("✓ Parameter ranges are reasonable")
    print(f"✓ Search space contains {len(search_space)} parameters")
    
    return True

if __name__ == '__main__':
    success = True
    success &= test_optuna_config()
    success &= test_search_space_coverage()
    
    if success:
        print("\n🎉 All tests passed! Optuna configuration is ready for use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        sys.exit(1)
