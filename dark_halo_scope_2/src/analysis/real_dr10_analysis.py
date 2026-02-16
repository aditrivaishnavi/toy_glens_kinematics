"""
Real DR10 analysis and known lens validation.

Apply trained detector to real data, cross-match with known lenses,
and search for new candidates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def apply_detector_to_dr10(
    model,
    backgrounds_h5: str,
    device,
    output_path: str = "data/processed/dr10_predictions.parquet",
    batch_size: int = 64
) -> pd.DataFrame:
    """
    Apply trained detector to all LRG backgrounds.
    
    Parameters
    ----------
    model : LensDetector
        Trained model
    backgrounds_h5 : str
        Path to preprocessed backgrounds HDF5
    device : torch.device
        Compute device
    output_path : str
        Output parquet path
    batch_size : int
        Batch size for inference
    
    Returns
    -------
    pd.DataFrame
        Predictions with columns: id, ra, dec, p_lens, theta_E_pred
    """
    import torch
    import h5py
    from tqdm import tqdm
    
    model.eval()
    
    all_ids = []
    all_ra = []
    all_dec = []
    all_probs = []
    all_theta_E = []
    
    with h5py.File(backgrounds_h5, 'r') as f:
        n_samples = f['ids'].shape[0]
        image_key = 'images_preproc' if 'images_preproc' in f else 'images_raw'
        
        for start in tqdm(range(0, n_samples, batch_size), desc="Applying to DR10"):
            end = min(start + batch_size, n_samples)
            
            images = torch.from_numpy(f[image_key][start:end]).to(device)
            
            with torch.no_grad():
                p_lens, theta_E_pred = model(images)
            
            all_ids.extend(f['ids'][start:end].tolist())
            all_ra.extend(f['ra'][start:end].tolist())
            all_dec.extend(f['dec'][start:end].tolist())
            all_probs.extend(p_lens.cpu().numpy().flatten().tolist())
            all_theta_E.extend(theta_E_pred.cpu().numpy().flatten().tolist())
    
    df = pd.DataFrame({
        'id': all_ids,
        'ra': all_ra,
        'dec': all_dec,
        'p_lens': all_probs,
        'theta_E_pred': all_theta_E
    })
    
    df = df.sort_values('p_lens', ascending=False).reset_index(drop=True)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    return df


def cross_match_known_lenses(
    predictions: pd.DataFrame,
    known_lens_catalog: pd.DataFrame,
    match_radius: float = 2.0
) -> pd.DataFrame:
    """
    Cross-match predictions with known lens catalog.
    
    Parameters
    ----------
    predictions : pd.DataFrame
        DR10 predictions with ra, dec columns
    known_lens_catalog : pd.DataFrame
        Known lenses with ra, dec, theta_E columns
    match_radius : float
        Match radius in arcseconds
    
    Returns
    -------
    pd.DataFrame
        Matched catalog with prediction info
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    # Build sky coordinates
    pred_coords = SkyCoord(
        ra=predictions['ra'].values * u.degree,
        dec=predictions['dec'].values * u.degree
    )
    
    known_coords = SkyCoord(
        ra=known_lens_catalog['ra'].values * u.degree,
        dec=known_lens_catalog['dec'].values * u.degree
    )
    
    # Cross-match
    idx, sep, _ = known_coords.match_to_catalog_sky(pred_coords)
    matched = sep < match_radius * u.arcsec
    
    # Build result
    result = known_lens_catalog.copy()
    result['pred_idx'] = idx
    result['separation_arcsec'] = sep.arcsec
    result['matched'] = matched
    
    # Add prediction info for matches
    result['p_lens'] = np.nan
    result['theta_E_pred'] = np.nan
    
    for i, row in result[matched].iterrows():
        pred_row = predictions.iloc[row['pred_idx']]
        result.loc[i, 'p_lens'] = pred_row['p_lens']
        result.loc[i, 'theta_E_pred'] = pred_row['theta_E_pred']
    
    return result


def compute_known_lens_recovery(
    matched: pd.DataFrame,
    threshold: float = 0.5
) -> Dict:
    """
    Compute recovery fraction of known lenses.
    
    Parameters
    ----------
    matched : pd.DataFrame
        Output from cross_match_known_lenses
    threshold : float
        Detection threshold
    
    Returns
    -------
    dict
        Recovery statistics
    """
    # Only consider matched objects
    matched_only = matched[matched['matched']]
    
    if len(matched_only) == 0:
        return {
            'n_known': len(matched),
            'n_matched': 0,
            'n_recovered': 0,
            'match_fraction': 0.0,
            'recovery_fraction': 0.0
        }
    
    # Recovered = matched AND detected above threshold
    recovered = matched_only['p_lens'] > threshold
    
    return {
        'n_known': len(matched),
        'n_matched': len(matched_only),
        'n_recovered': recovered.sum(),
        'match_fraction': len(matched_only) / len(matched),
        'recovery_fraction': recovered.sum() / len(matched_only) if len(matched_only) > 0 else 0
    }


def get_candidate_list(
    predictions: pd.DataFrame,
    n_top: int = 300,
    min_p_lens: float = 0.5
) -> pd.DataFrame:
    """
    Get top lens candidates for visual inspection.
    
    Parameters
    ----------
    predictions : pd.DataFrame
        DR10 predictions
    n_top : int
        Maximum number of candidates
    min_p_lens : float
        Minimum probability threshold
    
    Returns
    -------
    pd.DataFrame
        Top candidates sorted by probability
    """
    candidates = predictions[predictions['p_lens'] >= min_p_lens].copy()
    candidates = candidates.sort_values('p_lens', ascending=False)
    return candidates.head(n_top).reset_index(drop=True)


def grade_candidates(
    candidates: pd.DataFrame,
    grades: Dict[int, str]
) -> pd.DataFrame:
    """
    Add manual grades to candidate list.
    
    Parameters
    ----------
    candidates : pd.DataFrame
        Candidate list
    grades : dict
        Mapping from candidate index to grade ('A', 'B', 'C', 'reject')
    
    Returns
    -------
    pd.DataFrame
        Candidates with grade column
    """
    candidates = candidates.copy()
    candidates['grade'] = 'ungraded'
    
    for idx, grade in grades.items():
        if idx in candidates.index:
            candidates.loc[idx, 'grade'] = grade
    
    return candidates

