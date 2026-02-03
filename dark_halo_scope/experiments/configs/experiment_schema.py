"""
Experiment Configuration Schema with Validation.

This module defines the schema for reproducible experiment configurations.
All experiments MUST use this schema to ensure reproducibility.

Key principles:
1. Every random operation must have an explicit seed
2. All paths must be absolute or relative to a known root
3. Data variant must be explicitly specified
4. Model architecture and hyperparameters must be frozen at experiment start
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
import yaml
import json
import hashlib
from datetime import datetime, timezone


@dataclass
class SeedConfig:
    """Random seed configuration for reproducibility."""
    global_seed: int = 42
    numpy_seed: Optional[int] = None  # Defaults to global_seed if None
    torch_seed: Optional[int] = None  # Defaults to global_seed if None
    cuda_deterministic: bool = True
    
    def __post_init__(self):
        if self.numpy_seed is None:
            self.numpy_seed = self.global_seed
        if self.torch_seed is None:
            self.torch_seed = self.global_seed


@dataclass
class DataVariantConfig:
    """Data variant specification - tracks exactly which data is used."""
    variant_name: str  # e.g., "v4_sota_moffat"
    description: str
    phase3_parent_sample: str  # Path or S3 URI
    phase4a_manifest: str  # Path or S3 URI  
    phase4c_stamps: str  # Path or S3 URI
    
    # Injection parameters frozen at data generation
    psf_model: Literal["gaussian", "moffat"] = "gaussian"
    moffat_beta: Optional[float] = None
    source_mode: Literal["parametric", "cosmos"] = "parametric"
    cosmos_library_path: Optional[str] = None
    
    # Grid parameters
    theta_e_range: List[float] = field(default_factory=lambda: [0.5, 2.5])
    src_dmag_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
    
    # Control type
    control_type: Literal["paired", "unpaired"] = "unpaired"
    
    # Stamp parameters
    stamp_size: int = 64
    bands: List[str] = field(default_factory=lambda: ["g", "r", "z"])


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: Literal["resnet18", "convnext_tiny", "convnext_small"] = "convnext_tiny"
    pretrained: bool = False
    input_channels: int = 3
    dropout: float = 0.1
    
    # Metadata fusion
    use_metadata: bool = True
    metadata_columns: List[str] = field(default_factory=lambda: ["psfsize_r", "psfdepth_r"])
    metadata_hidden_dim: int = 64


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    
    # Optimizer
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    
    # Loss
    loss_type: Literal["bce", "focal"] = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Scheduler
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    warmup_epochs: int = 2
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_metric: str = "tpr_at_fpr1e-4"
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    
    # Data loading
    num_workers: int = 8
    prefetch_factor: int = 2
    
    # Augmentation
    augment: bool = True
    augment_flip: bool = True
    augment_rotate: bool = True
    
    # Filtering
    min_theta_over_psf: float = 0.5
    min_arc_snr: Optional[float] = None
    
    # Normalization
    norm_method: Literal["full", "outer", "percentile"] = "outer"


@dataclass
class HardNegativeConfig:
    """Hard negative mining configuration."""
    enabled: bool = False
    hard_neg_path: Optional[str] = None
    hard_neg_weight: float = 5.0
    min_score_threshold: float = 0.9
    
    # Real hard negative sources
    use_real_hard_negatives: bool = False
    ring_galaxy_catalog: Optional[str] = None
    merger_catalog: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    fpr_targets: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3, 1e-2])
    tpr_targets: List[float] = field(default_factory=lambda: [0.80, 0.85, 0.90, 0.95])
    
    # Stratification bins
    theta_e_bins: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5])
    theta_over_psf_bins: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.9, 1.1, 1.4, 2.0])
    
    # Anchor validation
    anchor_catalogs: List[str] = field(default_factory=list)
    match_radius_arcsec: float = 2.0


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.
    
    This is the top-level config that should be saved with every experiment.
    It contains all information needed to reproduce the experiment.
    """
    # Experiment identification
    experiment_id: str
    experiment_name: str
    description: str
    generation: str  # e.g., "gen5", "gen6"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Nested configs
    seeds: SeedConfig = field(default_factory=SeedConfig)
    data: DataVariantConfig = field(default_factory=lambda: DataVariantConfig(
        variant_name="UNSET",
        description="UNSET",
        phase3_parent_sample="UNSET",
        phase4a_manifest="UNSET",
        phase4c_stamps="UNSET"
    ))
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hard_negatives: HardNegativeConfig = field(default_factory=HardNegativeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Paths
    output_dir: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""
    
    # Git tracking (auto-filled at runtime)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration consistency."""
        errors = []
        
        # Data variant must be set
        if self.data.variant_name == "UNSET":
            errors.append("data.variant_name must be set")
        
        # Moffat requires beta
        if self.data.psf_model == "moffat" and self.data.moffat_beta is None:
            errors.append("moffat_beta must be set when psf_model is 'moffat'")
        
        # COSMOS mode requires library path
        if self.data.source_mode == "cosmos" and not self.data.cosmos_library_path:
            errors.append("cosmos_library_path must be set when source_mode is 'cosmos'")
        
        # Hard negatives require path if enabled
        if self.hard_negatives.enabled and not self.hard_negatives.hard_neg_path:
            errors.append("hard_neg_path must be set when hard_negatives.enabled is True")
        
        # Output directory must be set
        if not self.output_dir:
            errors.append("output_dir must be set")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    def compute_hash(self) -> str:
        """Compute a deterministic hash of the configuration for tracking."""
        # Convert to dict, excluding runtime fields
        d = asdict(self)
        for key in ['created_at', 'git_commit', 'git_branch', 'git_dirty']:
            d.pop(key, None)
        
        # Sort and hash
        json_str = json.dumps(d, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:12]
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Reconstruct nested dataclasses
        return cls(
            experiment_id=data['experiment_id'],
            experiment_name=data['experiment_name'],
            description=data['description'],
            generation=data['generation'],
            created_at=data.get('created_at', datetime.now(timezone.utc).isoformat()),
            seeds=SeedConfig(**data['seeds']),
            data=DataVariantConfig(**data['data']),
            model=ModelConfig(**data['model']),
            training=TrainingConfig(**data['training']),
            hard_negatives=HardNegativeConfig(**data['hard_negatives']),
            evaluation=EvaluationConfig(**data['evaluation']),
            output_dir=data['output_dir'],
            checkpoint_dir=data.get('checkpoint_dir', ''),
            log_dir=data.get('log_dir', ''),
            git_commit=data.get('git_commit'),
            git_branch=data.get('git_branch'),
            git_dirty=data.get('git_dirty')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_reproducibility_info(self) -> Dict[str, Any]:
        """Get summary of reproducibility-critical settings."""
        return {
            'config_hash': self.compute_hash(),
            'experiment_id': self.experiment_id,
            'generation': self.generation,
            'data_variant': self.data.variant_name,
            'seeds': {
                'global': self.seeds.global_seed,
                'numpy': self.seeds.numpy_seed,
                'torch': self.seeds.torch_seed,
                'cuda_deterministic': self.seeds.cuda_deterministic
            },
            'model': {
                'architecture': self.model.architecture,
                'dropout': self.model.dropout
            },
            'training': {
                'epochs': self.training.epochs,
                'batch_size': self.training.batch_size,
                'lr': self.training.learning_rate,
                'loss': self.training.loss_type
            },
            'git_commit': self.git_commit
        }


def set_all_seeds(config: SeedConfig) -> None:
    """Set all random seeds from configuration."""
    import random
    import numpy as np
    
    random.seed(config.global_seed)
    np.random.seed(config.numpy_seed)
    
    try:
        import torch
        torch.manual_seed(config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.torch_seed)
            if config.cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_info() -> Dict[str, Any]:
    """Get current git commit info for reproducibility tracking."""
    import subprocess
    
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Check if working directory is dirty
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = len(status) > 0
        
        return {'commit': commit, 'branch': branch, 'dirty': dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit': None, 'branch': None, 'dirty': None}


def create_experiment_config(
    experiment_name: str,
    generation: str,
    data_variant: DataVariantConfig,
    output_base_dir: str,
    description: str = "",
    **overrides
) -> ExperimentConfig:
    """
    Factory function to create a new experiment configuration.
    
    This ensures proper initialization and git tracking.
    """
    import uuid
    
    experiment_id = f"{generation}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    output_dir = str(Path(output_base_dir) / experiment_id)
    checkpoint_dir = str(Path(output_dir) / "checkpoints")
    log_dir = str(Path(output_dir) / "logs")
    
    git_info = get_git_info()
    
    config = ExperimentConfig(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        description=description,
        generation=generation,
        data=data_variant,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        git_commit=git_info['commit'],
        git_branch=git_info['branch'],
        git_dirty=git_info['dirty']
    )
    
    # Apply any overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
    
    return config

