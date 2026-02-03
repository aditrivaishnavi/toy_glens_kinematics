"""
Experiments module for Dark Halo Scope.

Provides infrastructure for reproducible experiment tracking including:
- Configuration schema with validation
- Data variant documentation
- Experiment result storage
"""

from .configs import (
    ExperimentConfig,
    SeedConfig,
    DataVariantConfig,
    ModelConfig,
    TrainingConfig,
    HardNegativeConfig,
    EvaluationConfig,
    set_all_seeds,
    create_experiment_config
)

__all__ = [
    'ExperimentConfig',
    'SeedConfig', 
    'DataVariantConfig',
    'ModelConfig',
    'TrainingConfig',
    'HardNegativeConfig',
    'EvaluationConfig',
    'set_all_seeds',
    'create_experiment_config'
]

