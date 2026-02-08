"""
Validation utilities.

Key lessons from lessons_learned:
- L4.4: "Data has no NaN values" - FALSE, always check
- L4.6: "Train/val/test is 70/15/15" - FALSE, always verify
- L5.4: Not validating code before upload
- L7.1: NaN values in raw stamps
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml


def validate_file_exists(
    path: Union[str, Path],
    name: Optional[str] = None,
    raise_error: bool = True
) -> Dict[str, Any]:
    """
    Validate that a file exists and has non-zero size.
    
    Args:
        path: Path to check
        name: Human-readable name for error messages
        raise_error: If True, raise FileNotFoundError on failure
    
    Returns:
        Dict with 'valid', 'path', 'size_bytes', 'error' keys
    """
    path = Path(path)
    name = name or path.name
    
    result = {
        'valid': False,
        'path': str(path),
        'name': name,
        'size_bytes': 0,
        'error': None
    }
    
    if not path.exists():
        result['error'] = f"{name} does not exist: {path}"
        if raise_error:
            raise FileNotFoundError(result['error'])
        return result
    
    if not path.is_file():
        result['error'] = f"{name} is not a file: {path}"
        if raise_error:
            raise ValueError(result['error'])
        return result
    
    size = path.stat().st_size
    result['size_bytes'] = size
    
    if size == 0:
        result['error'] = f"{name} is empty: {path}"
        if raise_error:
            raise ValueError(result['error'])
        return result
    
    result['valid'] = True
    return result


def validate_dataframe(
    df,
    required_columns: List[str],
    name: str = "DataFrame",
    min_rows: int = 1,
    check_nan: bool = True,
    nan_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate a pandas DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: Columns that must exist
        name: Name for error messages
        min_rows: Minimum number of rows required
        check_nan: If True, check for NaN values
        nan_columns: Specific columns to check for NaN (if None, check all)
    
    Returns:
        Dict with validation results
    """
    result = {
        'valid': True,
        'name': name,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'errors': [],
        'warnings': []
    }
    
    # Check row count
    if len(df) < min_rows:
        result['valid'] = False
        result['errors'].append(f"Too few rows: {len(df)} < {min_rows}")
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        result['valid'] = False
        result['errors'].append(f"Missing columns: {missing_cols}")
    
    # Check for NaN values
    if check_nan:
        cols_to_check = nan_columns if nan_columns else df.columns.tolist()
        cols_to_check = [c for c in cols_to_check if c in df.columns]
        
        for col in cols_to_check:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                result['warnings'].append(f"Column '{col}' has {nan_count} NaN values")
    
    return result


def validate_config(
    config: Dict[str, Any],
    required_keys: List[str],
    schema: Optional[Dict[str, type]] = None
) -> Dict[str, Any]:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration to validate
        required_keys: Keys that must exist
        schema: Optional dict mapping keys to expected types
    
    Returns:
        Dict with validation results
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required keys
    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        result['valid'] = False
        result['errors'].append(f"Missing required keys: {missing_keys}")
    
    # Check types if schema provided
    if schema:
        for key, expected_type in schema.items():
            if key in config:
                value = config[key]
                if not isinstance(value, expected_type):
                    result['valid'] = False
                    result['errors'].append(
                        f"Key '{key}' has wrong type: expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
    
    return result


def validate_split_proportions(
    df,
    split_column: str,
    expected_proportions: Dict[str, float],
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Validate that data splits have expected proportions.
    
    From lessons_learned L4.6: "Train/val/test is 70/15/15" - was actually 26/39/35!
    
    Args:
        df: DataFrame with split column
        split_column: Name of column containing split labels
        expected_proportions: Dict like {'train': 0.70, 'val': 0.15, 'test': 0.15}
        tolerance: Allowed deviation from expected
    
    Returns:
        Dict with validation results
    """
    result = {
        'valid': True,
        'actual_proportions': {},
        'expected_proportions': expected_proportions,
        'errors': [],
        'warnings': []
    }
    
    if split_column not in df.columns:
        result['valid'] = False
        result['errors'].append(f"Split column '{split_column}' not found")
        return result
    
    total = len(df)
    if total == 0:
        result['valid'] = False
        result['errors'].append("DataFrame is empty")
        return result
    
    # Calculate actual proportions
    value_counts = df[split_column].value_counts()
    for split_name in expected_proportions:
        count = value_counts.get(split_name, 0)
        actual = count / total
        result['actual_proportions'][split_name] = actual
        
        expected = expected_proportions[split_name]
        deviation = abs(actual - expected)
        
        if deviation > tolerance:
            result['valid'] = False
            result['errors'].append(
                f"Split '{split_name}': expected {expected:.2%}, got {actual:.2%} "
                f"(deviation: {deviation:.2%} > tolerance {tolerance:.2%})"
            )
        elif deviation > tolerance / 2:
            result['warnings'].append(
                f"Split '{split_name}': {actual:.2%} is close to tolerance "
                f"(expected {expected:.2%})"
            )
    
    # Check for unexpected splits
    for split_name in value_counts.index:
        if split_name not in expected_proportions:
            result['warnings'].append(f"Unexpected split value: '{split_name}'")
    
    return result


def run_all_validations(validations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run multiple validations and aggregate results.
    
    Args:
        validations: List of validation result dicts
    
    Returns:
        Aggregated results with 'all_passed' flag
    """
    result = {
        'all_passed': True,
        'total_validations': len(validations),
        'passed': 0,
        'failed': 0,
        'errors': [],
        'warnings': [],
        'details': validations
    }
    
    for v in validations:
        if v.get('valid', False):
            result['passed'] += 1
        else:
            result['failed'] += 1
            result['all_passed'] = False
            result['errors'].extend(v.get('errors', []))
        result['warnings'].extend(v.get('warnings', []))
    
    return result
