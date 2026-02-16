"""
I/O utilities for Dark Halo Scope.

File loading, saving, and format conversion helpers.
"""

import numpy as np
import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import h5py


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, path: str, indent: int = 2) -> None:
    """Save dictionary to JSON file."""
    ensure_dir(Path(path).parent)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict, path: str) -> None:
    """Save dictionary to YAML file."""
    ensure_dir(Path(path).parent)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(path: str) -> Dict:
    """Load dictionary from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_numpy(data: np.ndarray, path: str, compress: bool = True) -> None:
    """Save numpy array."""
    ensure_dir(Path(path).parent)
    if compress:
        np.savez_compressed(path, data=data)
    else:
        np.save(path, data)


def load_numpy(path: str) -> np.ndarray:
    """Load numpy array."""
    if path.endswith('.npz'):
        return np.load(path)['data']
    return np.load(path)


def get_hdf5_summary(path: str) -> Dict[str, Any]:
    """Get summary of HDF5 file contents."""
    summary = {'datasets': {}, 'attrs': {}}
    
    with h5py.File(path, 'r') as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                summary['datasets'][name] = {
                    'shape': obj.shape,
                    'dtype': str(obj.dtype)
                }
        
        f.visititems(visitor)
        summary['attrs'] = dict(f.attrs)
    
    return summary


def print_hdf5_summary(path: str) -> None:
    """Print formatted HDF5 summary."""
    summary = get_hdf5_summary(path)
    
    print(f"\nHDF5 Summary: {path}")
    print("=" * 50)
    
    print("\nDatasets:")
    for name, info in summary['datasets'].items():
        print(f"  /{name}: {info['shape']} {info['dtype']}")
    
    if summary['attrs']:
        print("\nAttributes:")
        for key, val in summary['attrs'].items():
            print(f"  {key}: {val}")


def sample_from_hdf5(
    path: str,
    n_samples: int = 5,
    dataset: str = 'images_raw'
) -> np.ndarray:
    """Load random samples from HDF5 dataset."""
    with h5py.File(path, 'r') as f:
        n_total = f[dataset].shape[0]
        indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
        indices = np.sort(indices)  # HDF5 prefers sorted indices
        return f[dataset][indices]


class ExperimentLogger:
    """Simple experiment logging."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = ensure_dir(Path(log_dir))
        self.experiment_name = experiment_name
        self.log_path = self.log_dir / f"{experiment_name}.json"
        self.data = {
            'name': experiment_name,
            'metrics': [],
            'config': {}
        }
    
    def set_config(self, config: Dict) -> None:
        """Set experiment configuration."""
        self.data['config'] = config
        self._save()
    
    def log_metrics(self, epoch: int, metrics: Dict) -> None:
        """Log metrics for an epoch."""
        entry = {'epoch': epoch, **metrics}
        self.data['metrics'].append(entry)
        self._save()
    
    def _save(self) -> None:
        """Save log to disk."""
        save_json(self.data, str(self.log_path))
    
    def load(self) -> Dict:
        """Load existing log."""
        if self.log_path.exists():
            self.data = load_json(str(self.log_path))
        return self.data

