"""
Experiment tracking utilities.

Key principles from lessons_learned:
- Configs are IMMUTABLE after creation
- Track git commit, timestamp, all parameters
- Every experiment has unique ID with timestamp
"""

import os
import json
import yaml
import shutil
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def get_git_info() -> Dict[str, Any]:
    """Get current git state for reproducibility."""
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
        dirty = subprocess.call(
            ['git', 'diff', '--quiet'],
            stderr=subprocess.DEVNULL
        ) != 0
        
        # Get short description
        try:
            describe = subprocess.check_output(
                ['git', 'describe', '--always', '--dirty'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except subprocess.CalledProcessError:
            describe = commit[:8]
        
        return {
            'commit': commit,
            'branch': branch,
            'dirty': dirty,
            'describe': describe,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            'commit': 'unknown',
            'branch': 'unknown',
            'dirty': True,
            'describe': 'unknown',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic hash of config for deduplication."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


class ExperimentTracker:
    """
    Track experiment configuration, code, and outputs.
    
    Usage:
        tracker = ExperimentTracker(
            experiment_name="emr_negative_sampling",
            config={"n_samples": 510000, "ratio": 100}
        )
        
        tracker.log_start()
        # ... run experiment ...
        tracker.log_end(status="success", metrics={"n_output": 510000})
    """
    
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        base_dir: Optional[Path] = None,
        force: bool = False
    ):
        self.experiment_name = experiment_name
        self.config = config
        self.force = force
        
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / 'experiments'
        self.base_dir = Path(base_dir)
        
        # Create unique experiment ID with timestamp
        self.timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        self.config_hash = compute_config_hash(config)
        self.experiment_id = f"{self.timestamp}_{experiment_name}_{self.config_hash}"
        
        self.experiment_dir = self.base_dir / self.experiment_id
        self.logs_dir = self.experiment_dir / 'logs'
        self.outputs_dir = self.experiment_dir / 'outputs'
        
        self._initialized = False
    
    def initialize(self) -> Path:
        """
        Create experiment directory and save config.
        
        Returns experiment directory path.
        """
        if self._initialized:
            return self.experiment_dir
        
        # Check for existing experiment with same config
        if not self.force:
            existing = self._find_existing_experiment()
            if existing:
                raise ExperimentExistsError(
                    f"Experiment with same config already exists: {existing}\n"
                    f"Use force=True to create new experiment anyway."
                )
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Save config (IMMUTABLE - never edit after this)
        config_path = self.experiment_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=True)
        
        # Save git info
        git_info = get_git_info()
        git_path = self.experiment_dir / 'git_info.json'
        with open(git_path, 'w') as f:
            json.dump(git_info, f, indent=2)
        
        # Save run metadata
        self.metadata = {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'config_hash': self.config_hash,
            'created_at': self.timestamp,
            'git_commit': git_info['commit'],
            'git_dirty': git_info['dirty'],
            'status': 'initialized',
            'start_time': None,
            'end_time': None,
            'duration_seconds': None,
            'metrics': {}
        }
        self._save_metadata()
        
        self._initialized = True
        print(f"Experiment initialized: {self.experiment_dir}")
        
        return self.experiment_dir
    
    def _find_existing_experiment(self) -> Optional[Path]:
        """Find existing experiment with same config hash."""
        if not self.base_dir.exists():
            return None
        
        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            if self.config_hash in exp_dir.name:
                # Verify config actually matches
                config_path = exp_dir / 'config.yaml'
                if config_path.exists():
                    with open(config_path) as f:
                        existing_config = yaml.safe_load(f)
                    if compute_config_hash(existing_config) == self.config_hash:
                        return exp_dir
        return None
    
    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_path = self.experiment_dir / 'run_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def log_start(self):
        """Log experiment start."""
        if not self._initialized:
            self.initialize()
        
        self.metadata['status'] = 'running'
        self.metadata['start_time'] = datetime.utcnow().isoformat() + 'Z'
        self._save_metadata()
        print(f"Experiment started: {self.experiment_id}")
    
    def log_end(self, status: str = 'success', metrics: Optional[Dict] = None):
        """Log experiment end with status and metrics."""
        end_time = datetime.utcnow()
        self.metadata['status'] = status
        self.metadata['end_time'] = end_time.isoformat() + 'Z'
        
        if self.metadata['start_time']:
            start = datetime.fromisoformat(self.metadata['start_time'].rstrip('Z'))
            self.metadata['duration_seconds'] = (end_time - start).total_seconds()
        
        if metrics:
            self.metadata['metrics'].update(metrics)
        
        self._save_metadata()
        print(f"Experiment {status}: {self.experiment_id}")
        if metrics:
            print(f"  Metrics: {metrics}")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log additional metrics during run."""
        self.metadata['metrics'].update(metrics)
        self._save_metadata()
    
    def get_log_path(self, name: str) -> Path:
        """Get path for a log file."""
        return self.logs_dir / name
    
    def get_output_path(self, name: str) -> Path:
        """Get path for an output file."""
        return self.outputs_dir / name
    
    def copy_artifact(self, src: Path, name: Optional[str] = None) -> Path:
        """Copy an artifact to the outputs directory."""
        if name is None:
            name = src.name
        dst = self.outputs_dir / name
        shutil.copy2(src, dst)
        return dst


class ExperimentExistsError(Exception):
    """Raised when trying to create duplicate experiment without force."""
    pass


def create_experiment(
    name: str,
    config: Dict[str, Any],
    force: bool = False
) -> ExperimentTracker:
    """
    Convenience function to create and initialize an experiment.
    
    Args:
        name: Experiment name (e.g., "emr_negative_sampling")
        config: Configuration dictionary
        force: If True, create even if same config exists
    
    Returns:
        Initialized ExperimentTracker
    """
    tracker = ExperimentTracker(name, config, force=force)
    tracker.initialize()
    return tracker


def list_experiments(
    base_dir: Optional[Path] = None,
    name_filter: Optional[str] = None
) -> list:
    """List all experiments, optionally filtered by name."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / 'experiments'
    
    experiments = []
    if not base_dir.exists():
        return experiments
    
    for exp_dir in sorted(base_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue
        
        if name_filter and name_filter not in exp_dir.name:
            continue
        
        metadata_path = exp_dir / 'run_metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            experiments.append({
                'path': exp_dir,
                'metadata': metadata
            })
    
    return experiments
