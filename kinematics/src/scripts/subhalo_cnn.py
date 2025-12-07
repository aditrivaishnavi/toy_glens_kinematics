# src/scripts/subhalo_cnn.py
"""
Train a CNN to distinguish:
  - smooth SIS lenses  (label 0)
  - SIS + on-arc SIS subhalo lenses (label 1)

This script is the "subhalo experiment" and is intentionally separate from the
simpler lens-vs-unlensed experiment (cnn_lensed_vs_unlensed.py).

================================================================================
STEP-2 EXPERIMENT DESIGN
================================================================================

Goal: "Can the CNN see a clear subhalo signal when we make the physics easier?"

This step deliberately uses:
  - STRONG subhalo (theta_E_sub_factor ~ 0.3) to produce obvious perturbations
  - ZERO or LOW PSF/noise initially, then gradually increased
  - THREE channel modes for ablation: flux_only, vel_only, both

Expected outcomes:
  1. Clean case (no PSF/noise): CNN should reach >90% accuracy
  2. As PSF/noise increases, accuracy should degrade
  3. Comparing flux_only vs vel_only vs both reveals which modality carries
     the subhalo signal

================================================================================
PHYSICS KNOBS (all exposed as CLI args)
================================================================================

--theta_E_main        : Main lens Einstein radius (default 0.5)
--theta_E_sub_factor  : Subhalo strength as fraction of main (default 0.3 for Step-2)
--psf_sigma           : Gaussian PSF sigma in pixels (default 0.0 = no PSF)
--flux_noise_sigma    : Flux channel noise sigma (default 0.0 = no noise)
--vel_noise_sigma     : Velocity channel noise sigma (default 0.0 = no noise)
--flux_mask_threshold : Below this flux, velocity is masked to 0 (default 0.1)
--channel_mode        : 'both', 'flux_only', or 'vel_only'

================================================================================
USAGE EXAMPLES
================================================================================

# Step 2a: Clean experiment (no PSF, no noise), strong subhalo
python src/scripts/subhalo_cnn.py \\
    --theta_E_sub_factor 0.3 \\
    --psf_sigma 0.0 \\
    --flux_noise_sigma 0.0 \\
    --vel_noise_sigma 0.0 \\
    --channel_mode both \\
    --epochs 10

# Step 2b: Compare flux-only
python src/scripts/subhalo_cnn.py \\
    --theta_E_sub_factor 0.3 \\
    --channel_mode flux_only \\
    --output_model models/subhalo_cnn_flux_only.pt

# Step 2c: Compare vel-only
python src/scripts/subhalo_cnn.py \\
    --theta_E_sub_factor 0.3 \\
    --channel_mode vel_only \\
    --output_model models/subhalo_cnn_vel_only.pt

# Step 2d: Add moderate PSF + noise
python src/scripts/subhalo_cnn.py \\
    --theta_E_sub_factor 0.3 \\
    --psf_sigma 1.0 \\
    --flux_noise_sigma 0.08 \\
    --vel_noise_sigma 0.03 \\
    --output_model models/subhalo_cnn_with_noise.pt

================================================================================
IMPLEMENTATION NOTES
================================================================================

IMPORTANT:
- All lens + subhalo + PSF + noise + masking logic is delegated to
  src.glens.lensing_utils, which is also used by debug_subhalo_samples.py.
- This ensures the training distribution matches the debug plots exactly.

TODO (future milestones, NOT for this step):
- Replace SIS subhalo with truncated NFW using lenstronomy.
- Move from fixed SIS main lens to elliptic SIE + external shear.
- Curriculum learning: start noise-free, then increase noise.
- Deeper architectures (ResNet) once basic signal is detectable.
"""

import argparse
import os
import sys
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.glens.lensing_utils import (
    LensSimConfig,
    load_source_tensor,
    render_sis_lens,
    render_sis_plus_subhalo,
    apply_psf_and_noise,
)


# ------------------------------
# Small CNN architecture
# ------------------------------

class SmallCNN(nn.Module):
    """
    Minimal 2D CNN for binary classification.

    Input: (batch, C, H, W) where C=2 channels.
           Even in flux_only / vel_only modes we still keep 2 channels
           (one of them zero) so the architecture is stable.

    Architecture:
        Conv2d(2, 16, 3) -> ReLU -> MaxPool(2)
        Conv2d(16, 32, 3) -> ReLU -> AdaptiveAvgPool(1)
        Linear(32, 2)

    NOTE: For future, we may swap this to a deeper architecture (e.g., ResNet)
    once the basic signal is clearly detectable with this simple model.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ------------------------------
# Dataset
# ------------------------------

class SubhaloLensDataset(Dataset):
    """
    Synthetic dataset of (flux, velocity) lensed images.

    Each sample:
      - label 0: smooth SIS lens (no subhalo)
      - label 1: SIS + on-arc SIS subhalo

    The exact simulation pipeline is shared with debug_subhalo_samples.py via
    src.glens.lensing_utils. This guarantees that the training data matches the
    debug plots in physics and noise properties.

    Channel modes:
      - 'both'      -> [flux, vel]  (standard 2-channel input)
      - 'flux_only' -> [flux, 0]    (ablation: can CNN detect from morphology alone?)
      - 'vel_only'  -> [0, vel]     (ablation: can CNN detect from kinematics alone?)

    Step-2 experiment design:
      - theta_E_sub_factor ~ 0.3 for strong, clearly visible perturbation
      - PSF/noise can be set to 0.0 for clean experiment, then gradually increased
      - Compare flux_only vs vel_only vs both to identify which modality matters

    NOTE:
    - For this Step-2 experiment, we intentionally make the subhalo relatively
      strong and allow turning PSF/noise off completely.
    - Later steps will gradually reduce subhalo strength and increase PSF/noise
      to explore more realistic (harder) regimes.
    """

    def __init__(
        self,
        plateifus: List[str],
        tensor_dir: str,
        n_samples: int,
        config: LensSimConfig,
        channel_mode: str = "both",
        seed: int = 0,
    ):
        super().__init__()
        self.plateifus = plateifus
        self.tensor_dir = tensor_dir
        self.n_samples = n_samples
        self.config = config
        self.channel_mode = channel_mode
        self.rng = np.random.default_rng(seed)

        assert channel_mode in ("both", "flux_only", "vel_only"), \
            f"channel_mode must be 'both', 'flux_only', or 'vel_only', got {channel_mode}"

        # Cache loaded tensors for efficiency
        self._cache = {}

    def __len__(self) -> int:
        return self.n_samples

    def _get_source(self, plateifu: str) -> np.ndarray:
        """Load and cache source tensor."""
        if plateifu not in self._cache:
            self._cache[plateifu] = load_source_tensor(plateifu, self.tensor_dir)
        return self._cache[plateifu].copy()

    def _augment(self, tensor: np.ndarray) -> np.ndarray:
        """Apply random rotations and flips (same for both channels)."""
        arr = tensor.copy()
        # Random 90-degree rotations
        k = self.rng.integers(0, 4)
        if k > 0:
            arr = np.rot90(arr, k=k, axes=(1, 2))
        # Random horizontal flip
        if self.rng.random() < 0.5:
            arr = arr[:, :, ::-1].copy()
        # Random vertical flip
        if self.rng.random() < 0.5:
            arr = arr[:, ::-1, :].copy()
        return arr

    def __getitem__(self, idx: int):
        # Randomly choose which galaxy to use
        plateifu = self.plateifus[self.rng.integers(0, len(self.plateifus))]

        # Load source tensor (2, H, W), already normalized & resampled
        source = self._get_source(plateifu)
        source = self._augment(source)

        # Binary label: 0 = smooth, 1 = subhalo
        label = self.rng.integers(0, 2)

        if label == 0:
            # Smooth SIS lens (no subhalo)
            lensed = render_sis_lens(source, config=self.config)
        else:
            # SIS + subhalo (on-arc)
            lensed = render_sis_plus_subhalo(
                source, config=self.config, on_arc=True, rng=self.rng
            )

        # Apply PSF + noise (if sigma > 0)
        # Note: apply_psf_and_noise handles sigma=0 gracefully (no-op for PSF)
        lensed = apply_psf_and_noise(lensed, config=self.config, rng=self.rng)

        # Extract flux and velocity channels
        flux = lensed[0]
        vel = lensed[1]

        # Build 2-channel tensor according to channel_mode
        if self.channel_mode == "both":
            stacked = np.stack([flux, vel], axis=0)
        elif self.channel_mode == "flux_only":
            zeros = np.zeros_like(flux)
            stacked = np.stack([flux, zeros], axis=0)
        else:  # vel_only
            zeros = np.zeros_like(vel)
            stacked = np.stack([zeros, vel], axis=0)

        # Convert to torch
        x = torch.from_numpy(stacked.astype(np.float32))
        y = torch.tensor(label, dtype=torch.long)

        return x, y


# ------------------------------
# Training / evaluation helpers
# ------------------------------

def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch, return (loss, accuracy)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model, return (loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, running_correct / total


# ------------------------------
# Main
# ------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN: smooth SIS vs SIS+subhalo (on-arc). Step-2 experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean experiment (no PSF/noise), strong subhalo:
  python src/scripts/subhalo_cnn.py --theta_E_sub_factor 0.3 --epochs 10

  # Flux-only ablation:
  python src/scripts/subhalo_cnn.py --channel_mode flux_only

  # With PSF + noise:
  python src/scripts/subhalo_cnn.py --psf_sigma 1.0 --flux_noise_sigma 0.08 --vel_noise_sigma 0.03
        """
    )

    # Data and training
    parser.add_argument("--tensor_dir", type=str, default="data/source_tensors")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_samples", type=int, default=800)
    parser.add_argument("--test_samples", type=int, default=800)
    parser.add_argument("--output_model", type=str, default="models/subhalo_cnn.pt")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Physics knobs (all passed into LensSimConfig)
    parser.add_argument("--theta_E_main", type=float, default=0.5,
                        help="Main SIS Einstein radius in normalized units.")
    parser.add_argument("--theta_E_sub_factor", type=float, default=0.3,
                        help="Subhalo Einstein radius as fraction of main. "
                             "Step-2 default: 0.3 (strong signal).")
    parser.add_argument("--psf_sigma", type=float, default=0.0,
                        help="Gaussian PSF sigma in pixels. "
                             "Step-2 initial: 0.0 (no PSF). Later: increase to 1.0.")
    parser.add_argument("--flux_noise_sigma", type=float, default=0.0,
                        help="Flux channel noise sigma. "
                             "Step-2 initial: 0.0. Later: increase gradually.")
    parser.add_argument("--vel_noise_sigma", type=float, default=0.0,
                        help="Velocity channel noise sigma. "
                             "Step-2 initial: 0.0. Later: increase gradually.")
    parser.add_argument("--flux_mask_threshold", type=float, default=0.1,
                        help="Flux threshold for velocity masking.")

    # Channel mode for ablation studies
    parser.add_argument(
        "--channel_mode",
        type=str,
        default="both",
        choices=["both", "flux_only", "vel_only"],
        help="Which channels to feed to the CNN. "
             "'both' = [flux, vel], 'flux_only' = [flux, 0], 'vel_only' = [0, vel]"
    )

    # Random seeds
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    # Fixed plateifu split (same as previous experiments)
    train_plateifus = ["8593-12705", "8993-12705", "11982-9102", "10500-12703"]
    val_plateifus = ["8652-12703"]
    test_plateifus = ["9487-9102"]

    print("============================================================")
    print("SUBHALO CNN TRAINING (Step-2: smooth vs SIS+subhalo)")
    print("============================================================")
    print(f"[INFO] tensor_dir      : {args.tensor_dir}")
    print(f"[INFO] train plateifus : {train_plateifus}")
    print(f"[INFO] val plateifus   : {val_plateifus}")
    print(f"[INFO] test plateifus  : {test_plateifus}")
    print(f"[INFO] batch_size      : {args.batch_size}")
    print(f"[INFO] epochs          : {args.epochs}")
    print(f"[INFO] learning_rate   : {args.lr}")
    print(f"[INFO] output_model    : {args.output_model}")
    print(f"[INFO] channel_mode    : {args.channel_mode}")
    print("---- Physics config (LensSimConfig) ----")
    print(f"  theta_E_main        = {args.theta_E_main}")
    print(f"  theta_E_sub_factor  = {args.theta_E_sub_factor}")
    print(f"  psf_sigma           = {args.psf_sigma}")
    print(f"  flux_noise_sigma    = {args.flux_noise_sigma}")
    print(f"  vel_noise_sigma     = {args.vel_noise_sigma}")
    print(f"  flux_mask_threshold = {args.flux_mask_threshold}")
    print("============================================================")

    # Seed everything for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build config from CLI args
    config = LensSimConfig(
        theta_E_main=args.theta_E_main,
        theta_E_sub_factor=args.theta_E_sub_factor,
        psf_sigma=args.psf_sigma,
        flux_noise_sigma=args.flux_noise_sigma,
        vel_noise_sigma=args.vel_noise_sigma,
        flux_mask_threshold=args.flux_mask_threshold,
    )

    # Create datasets
    train_dataset = SubhaloLensDataset(
        plateifus=train_plateifus,
        tensor_dir=args.tensor_dir,
        n_samples=args.train_samples,
        config=config,
        channel_mode=args.channel_mode,
        seed=args.seed,
    )
    val_dataset = SubhaloLensDataset(
        plateifus=val_plateifus,
        tensor_dir=args.tensor_dir,
        n_samples=args.val_samples,
        config=config,
        channel_mode=args.channel_mode,
        seed=args.seed + 1,
    )
    test_dataset = SubhaloLensDataset(
        plateifus=test_plateifus,
        tensor_dir=args.tensor_dir,
        n_samples=args.test_samples,
        config=config,
        channel_mode=args.channel_mode,
        seed=args.seed + 2,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Model, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = SmallCNN(in_channels=2, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict().copy()

    # Load best model for final test
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, device)

    print("============================================================")
    print(f"[RESULT] Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.3f}")
    print("============================================================")

    # Interpretation guidance
    if test_acc > 0.9:
        print("[INFO] Excellent! CNN clearly detects the subhalo signal.")
        print("       Try increasing PSF/noise or reducing theta_E_sub_factor.")
    elif test_acc > 0.7:
        print("[INFO] Good detection. Signal is visible but challenging.")
    elif test_acc > 0.55:
        print("[INFO] Weak detection. Consider stronger subhalo or less noise.")
    else:
        print("[INFO] Near chance level. Subhalo signal may be too weak or buried in noise.")

    # Save model
    torch.save(model.state_dict(), args.output_model)
    print(f"[OK] Saved model state_dict to: {args.output_model}")


if __name__ == "__main__":
    main()
