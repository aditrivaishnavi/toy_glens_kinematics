"""
Plotting utilities for Dark Halo Scope.

Standard visualization functions for observability maps,
selection functions, and detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, Dict


# Custom colormap for selection function
SELECTION_CMAP = LinearSegmentedColormap.from_list(
    'selection',
    ['#2c3e50', '#3498db', '#2ecc71', '#f1c40f']
)


def plot_observability_map(
    theta_E_grid: np.ndarray,
    z_l_grid: np.ndarray,
    classification: np.ndarray,
    FWHM: float,
    ax: Optional[plt.Axes] = None,
    title: str = "Observability Map"
) -> plt.Axes:
    """
    Plot observability classification in θ_E–z_l space.
    
    Parameters
    ----------
    theta_E_grid, z_l_grid : np.ndarray
        1D arrays of θ_E and z_l values
    classification : np.ndarray
        2D array (n_theta, n_z) with values 0=blind, 1=low_trust, 2=good
    FWHM : float
        Seeing FWHM for annotation
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map: red=blind, yellow=low_trust, green=good
    cmap = plt.cm.RdYlGn
    
    im = ax.pcolormesh(
        z_l_grid, theta_E_grid, classification,
        cmap=cmap, vmin=0, vmax=2, shading='auto'
    )
    
    # Add contour lines
    cs = ax.contour(
        z_l_grid, theta_E_grid, classification,
        levels=[0.5, 1.5], colors='k', linewidths=1.5
    )
    
    # Labels
    ax.set_xlabel(r'Lens Redshift $z_l$', fontsize=12)
    ax.set_ylabel(r'Einstein Radius $\theta_E$ (arcsec)', fontsize=12)
    ax.set_title(f'{title}\n(FWHM = {FWHM:.2f}")', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0.33, 1, 1.67])
    cbar.ax.set_yticklabels(['Blind', 'Low Trust', 'Good'])
    
    return ax


def plot_selection_function(
    selection_dict: Dict,
    ax: Optional[plt.Axes] = None,
    title: str = "Selection Function C(θ_E, z_l)",
    show_blind_boundary: bool = True
) -> plt.Axes:
    """
    Plot selection function heatmap.
    
    Parameters
    ----------
    selection_dict : dict
        Output from build_selection_function_grid
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title
    show_blind_boundary : bool
        Whether to show analytic blind boundary
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    selection = selection_dict['selection']
    theta_E = selection_dict['theta_E_centers']
    z_l = selection_dict['z_l_centers']
    
    im = ax.pcolormesh(
        z_l, theta_E, selection,
        cmap=SELECTION_CMAP, vmin=0, vmax=1, shading='auto'
    )
    
    # Blind boundary
    if show_blind_boundary:
        blind_mask = selection_dict['analytic_class'] == 0
        ax.contour(
            z_l, theta_E, blind_mask.astype(float),
            levels=[0.5], colors='red', linewidths=2,
            linestyles='--'
        )
    
    ax.set_xlabel(r'Lens Redshift $z_l$', fontsize=12)
    ax.set_ylabel(r'Einstein Radius $\theta_E$ (arcsec)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Completeness C', fontsize=11)
    
    return ax


def plot_completeness_curves(
    theta_E_centers: np.ndarray,
    completeness: np.ndarray,
    counts: np.ndarray,
    ax: Optional[plt.Axes] = None,
    label: str = "",
    color: str = 'blue'
) -> plt.Axes:
    """
    Plot completeness vs θ_E curve.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Only plot where we have enough samples
    valid = counts >= 5
    
    ax.plot(theta_E_centers[valid], completeness[valid], 
            'o-', color=color, label=label, linewidth=2)
    
    ax.set_xlabel(r'Einstein Radius $\theta_E$ (arcsec)', fontsize=12)
    ax.set_ylabel('Completeness', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    if label:
        ax.legend()
    
    return ax


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_value: float,
    ax: Optional[plt.Axes] = None,
    label: str = "Detector"
) -> plt.Axes:
    """
    Plot ROC curve.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {auc_value:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return ax


def plot_rgb_cutout(
    cutout: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    scale: str = 'linear'
) -> plt.Axes:
    """
    Plot RGB composite of a cutout.
    
    Parameters
    ----------
    cutout : np.ndarray
        Shape (3, H, W) or (H, W, 3) in g, r, z order
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title
    scale : str
        'linear' or 'asinh' scaling
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Ensure (H, W, 3) format
    if cutout.shape[0] == 3:
        img = np.transpose(cutout, (1, 2, 0))
    else:
        img = cutout
    
    # Normalize per channel
    rgb = np.zeros_like(img)
    for i in range(3):
        channel = img[:, :, i]
        vmin = np.nanpercentile(channel, 1)
        vmax = np.nanpercentile(channel, 99)
        if vmax > vmin:
            if scale == 'asinh':
                rgb[:, :, i] = np.arcsinh((channel - vmin) / (vmax - vmin) * 10) / np.arcsinh(10)
            else:
                rgb[:, :, i] = np.clip((channel - vmin) / (vmax - vmin), 0, 1)
    
    # Swap to RGB order (assuming input is g, r, z)
    # Display as r, g, b → use z, r, g
    display = rgb[:, :, ::-1]  # z, r, g
    
    ax.imshow(display, origin='lower')
    ax.set_title(title)
    ax.axis('off')
    
    return ax


def plot_injection_gallery(
    images: np.ndarray,
    params_list: list,
    n_cols: int = 4,
    figsize: Tuple[float, float] = (16, 12)
) -> plt.Figure:
    """
    Plot gallery of injected lenses.
    
    Parameters
    ----------
    images : np.ndarray
        Shape (N, 3, H, W) images
    params_list : list
        List of InjectionParams or dicts with θ_E, z_l, etc.
    n_cols : int
        Number of columns
    figsize : tuple
        Figure size
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (img, params) in enumerate(zip(images, params_list)):
        ax = axes[i]
        plot_rgb_cutout(img, ax=ax)
        
        if isinstance(params, dict):
            title = f"θ_E={params.get('theta_E', 0):.2f}\""
        else:
            title = f"θ_E={params.theta_E:.2f}\""
        ax.set_title(title, fontsize=10)
    
    # Hide empty axes
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

