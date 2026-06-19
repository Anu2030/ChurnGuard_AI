import os
import sys
import pytest

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def test_config_paths_exist():
    """Test that the base directories defined in config exist."""
    # We only check directories, because files might not be generated yet
    assert os.path.exists(config.BASE_DIR)
    
def test_hyperparameter_grids_are_valid():
    """Test that the XGBoost and RF grids are properly formatted dicts."""
    assert isinstance(config.GRID_PARAMS_XGB, dict)
    assert 'n_estimators' in config.GRID_PARAMS_XGB
    assert 'learning_rate' in config.GRID_PARAMS_XGB
    
    assert isinstance(config.GRID_PARAMS_RF, dict)
    assert 'max_depth' in config.GRID_PARAMS_RF

def test_feature_lists_are_valid():
    """Test that numeric and categorical feature lists are populated."""
    assert isinstance(config.NUMERIC_COLS, list)
    assert len(config.NUMERIC_COLS) > 0
    assert 'MonthlyCharges' in config.NUMERIC_COLS
    
    assert isinstance(config.CATEGORICAL_COLS, list)
    assert len(config.CATEGORICAL_COLS) > 0
    assert 'Contract' in config.CATEGORICAL_COLS
