"""
Experiment configuration module.

Provides schema definitions and utilities for reproducible experiment tracking.
"""

from .experiment_schema import (
    ExperimentConfig,
    SeedConfig,
    DataVariantConfig,
    ModelConfig,
    TrainingConfig,
    HardNegativeConfig,
    EvaluationConfig,
    set_all_seeds,
    get_git_info,
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
    'get_git_info',
    'create_experiment_config'
]

