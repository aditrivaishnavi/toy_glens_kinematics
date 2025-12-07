#!/usr/bin/env python
"""
toy_cnn.py

Milestone 1 CNN prototype for the MaNGA + toy lensing project.

Goal
-----
Train a small 2-channel CNN to distinguish:
  class 0: UNLENSED source maps
  class 1: LENSED maps (toy SIS lens)

using the normalized 2-channel tensors produced by `prep_source_maps.py`:
  data/source_tensors/source_tensor_<plateifu>.npy

This is an *end-to-end plumbing test*:
- Validates that:
    - our (flux, velocity) tensors are well-formed;
    - our toy SIS lensing works numerically;
    - a CNN can extract a strong signal from the 2-channel maps.
- Does NOT yet:
    - inject realistic subhalos;
    - model PSF, noise, JWST specifics;
    - estimate subhalo mass/position.

Design decisions (for now)
--------------------------
- Resolution: 64 x 64 (as in prep_source_maps.py).
- Channels:
    channel 0 = flux in [0, 1]
    channel 1 = normalized velocity ~ [-1, 1]
- Lensing:
    simple SIS lens at the origin with random Einstein radius
    theta_E ~ Uniform[0.3, 0.7] (in normalized [-1,1] coordinates)
- Masking:
    velocity is zeroed outside regions where lensed flux >= flux_mask_thresh.
- Data generation:
    lensed vs unlensed examples generated ON THE FLY per batch; no pre-saved
    lensed tensors are required.

Future TODOs (Milestone 2+)
---------------------------
- Replace this internal SIS implementation with a shared module used by toy_lens.py.
- Add a "with-subhalo vs smooth" classification task, with guaranteed on-arc
  subhalo placement.
- Introduce PSF, noise, and higher-resolution sampling for sub-pixel effects.
- Externalize train/val/test splits into index files instead of hardcoding
  plateifu lists below.

IMPORTANT CAVEAT
----------------
This toy CNN task is intentionally easy (strong lens vs unlensed). Achieving
~100% accuracy only shows:
  1. The pipeline is wired correctly end-to-end.
  2. The lensing transform is not numerically broken.
  3. The CNN can detect obvious morphological differences (ring vs blob).

It does NOT establish performance on realistic subhalo perturbations, which
produce much subtler signals. High accuracy here is a sanity check, not a
scientific result.

Diagnostic TODOs
----------------
TODO: Check class balance per split
      Log counts of class 0 vs 1 in train/val/test after dataset creation.
      Should be ~50/50 by construction (lens_probability=0.5).

TODO: Quick ablation - flux-only vs vel-only
      Train the same CNN architecture on:
        - Only flux channel (1 channel input)
        - Only velocity channel (1 channel input)
      Compare accuracies. If flux-only already gives ~100% and vel-only is
      much lower, we know the velocity channel isn't needed for this toy task.
      This would confirm the task is "ring detection" not "kinematic anomaly".

TODO: Baseline classifier (non-CNN)
      Compute simple scalar features:
        - central_flux = mean(flux[center 5x5])
        - ring_flux = mean(flux[annulus r=20..30 pixels])
        - ratio = ring_flux / (central_flux + eps)
      Train a logistic regression or simple threshold rule on these features.
      If that also gives ~100%, then we've confirmed: this problem is trivial,
      and the CNN is just a fancy way of detecting ring vs blob morphology.

"""

import argparse
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Configuration: plateifu splits
# ---------------------------------------------------------------------
#
# NOTE: These splits are *by galaxy*, to avoid leakage. Each plateifu
# appears in exactly one of (train, val, test).
#
# TODO (Milestone 2+): Load these from external index files such as
#   data/index_train_plateifus.txt
#   data/index_val_plateifus.txt
#   data/index_test_plateifus.txt

TRAIN_PLATEIFUS = [
    "8593-12705",
    "8993-12705",
    "11982-9102",
    "10500-12703",
]

VAL_PLATEIFUS = [
    "8652-12703",
]

TEST_PLATEIFUS = [
    "9487-9102",
]


# ---------------------------------------------------------------------
# Simple SIS lensing (duplicated from toy_lens.py for now)
# ---------------------------------------------------------------------
#
# NOTE: This is intentionally minimal and self-contained so that
# toy_cnn.py can run without import-path headaches. In a later refactor,
# factor this into a shared module and have both toy_lens.py and
# toy_cnn.py import from there.
#

def apply_sis_lens(
    source_tensor: np.ndarray,
    theta_E: float = 0.5,
    flux_mask_threshold: float = 0.1,
) -> np.ndarray:
    """
    Apply a simple SIS lens at the origin to a (2, H, W) source tensor.

    Parameters
    ----------
    source_tensor : np.ndarray
        Shape (2, H, W), channels = (flux, velocity), already normalized.
    theta_E : float
        Einstein radius in normalized image-plane coordinates [-1,1]x[-1,1].
    flux_mask_threshold : float
        Flux threshold in [0,1] above which we consider velocity to be
        physically meaningful. Pixels below this are zeroed in the lensed
        velocity map.

    Returns
    -------
    np.ndarray
        Lensed tensor of shape (2, H, W).
    """
    assert source_tensor.ndim == 3 and source_tensor.shape[0] == 2

    _, H, W = source_tensor.shape
    flux_src = source_tensor[0]
    vel_src = source_tensor[1]

    # Build image-plane coordinate grid in [-1,1] x [-1,1]
    y = np.linspace(-1.0, 1.0, H)
    x = np.linspace(-1.0, 1.0, W)
    theta_y, theta_x = np.meshgrid(y, x, indexing="ij")  # shape (H, W)

    # SIS deflection
    r = np.sqrt(theta_x**2 + theta_y**2) + 1e-6  # avoid div by zero
    alpha_x = theta_E * theta_x / r
    alpha_y = theta_E * theta_y / r

    beta_x = theta_x - alpha_x
    beta_y = theta_y - alpha_y

    # Map (beta_x, beta_y) back to source pixel coordinates
    # normalized [-1,1] -> pixel indices [0..W-1], [0..H-1]
    src_x = (beta_x + 1.0) * 0.5 * (W - 1)
    src_y = (beta_y + 1.0) * 0.5 * (H - 1)

    # Bilinear sampling for flux and velocity
    def bilinear_sample(img: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
        H_, W_ = img.shape
        x0 = np.floor(x_idx).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y_idx).astype(int)
        y1 = y0 + 1

        # clip to image bounds
        x0 = np.clip(x0, 0, W_ - 1)
        x1 = np.clip(x1, 0, W_ - 1)
        y0 = np.clip(y0, 0, H_ - 1)
        y1 = np.clip(y1, 0, H_ - 1)

        Ia = img[y0, x0]
        Ib = img[y1, x0]
        Ic = img[y0, x1]
        Id = img[y1, x1]

        wa = (x1 - x_idx) * (y1 - y_idx)
        wb = (x1 - x_idx) * (y_idx - y0)
        wc = (x_idx - x0) * (y1 - y_idx)
        wd = (x_idx - x0) * (y_idx - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    lensed_flux = bilinear_sample(flux_src, src_x, src_y)
    lensed_vel = bilinear_sample(vel_src, src_x, src_y)

    # Flux-mask the velocity: only trust velocity where flux is significant
    mask = lensed_flux >= flux_mask_threshold
    lensed_vel_masked = np.zeros_like(lensed_vel)
    lensed_vel_masked[mask] = lensed_vel[mask]

    out = np.zeros_like(source_tensor)
    out[0] = lensed_flux
    out[1] = lensed_vel_masked

    return out


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class ToyLensDataset(Dataset):
    """
    PyTorch Dataset for lensed vs unlensed 2-channel MaNGA tensors.

    Each __getitem__:
      - picks a random plateifu from the provided list,
      - optionally lenses it with SIS,
      - applies simple geometric augmentations,
      - returns (tensor, label), where:
          tensor: torch.FloatTensor of shape (2, H, W)
          label : 0 for UNLENSED, 1 for LENSED

    This is intentionally simple and CPU friendly. All heavy lifting
    (MaNGA I/O, normalization, resampling) was done in prep_source_maps.py.
    """

    def __init__(
        self,
        plateifus: List[str],
        tensor_dir: str,
        n_samples: int = 2000,
        lens_probability: float = 0.5,
        theta_E_range: Tuple[float, float] = (0.3, 0.7),
        flux_mask_threshold: float = 0.1,
        augment: bool = True,
        seed: int = 1234,
    ):
        super().__init__()
        self.plateifus = plateifus
        self.tensor_dir = tensor_dir
        self.n_samples = n_samples
        self.lens_probability = lens_probability
        self.theta_E_range = theta_E_range
        self.flux_mask_threshold = flux_mask_threshold
        self.augment = augment

        # RNG for reproducibility (dataset-level only)
        self.rng = np.random.default_rng(seed)

        # Load source tensors for each plateifu once into memory
        self.source_tensors: Dict[str, np.ndarray] = {}
        for pid in plateifus:
            path = os.path.join(tensor_dir, f"source_tensor_{pid}.npy")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing source tensor for plateifu {pid}: {path}"
                )
            arr = np.load(path)  # shape (2, H, W)
            if arr.shape[0] != 2:
                raise ValueError(f"Expected 2 channels for {pid}, got {arr.shape}")
            self.source_tensors[pid] = arr.astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def _random_augmentation(self, tensor: np.ndarray) -> np.ndarray:
        """
        Apply simple geometric + noise augmentations.

        All operations preserve the physical meaning in a rotationally
        invariant sense (i.e., the viewer does not know "which way is up").

        Currently:
          - random rotation by k * 90 degrees
          - random horizontal flip
          - small Gaussian noise (sigma ~ 0.01)

        TODO (Milestone 2+):
          - consider elastic or small affine distortions, but ensure we
            apply them identically to both channels.
        """
        img = tensor

        # Random 90-degree rotations
        k = self.rng.integers(0, 4)
        if k > 0:
            img = np.rot90(img, k=k, axes=(1, 2))  # rotate H,W axes

        # Random horizontal flip
        if self.rng.random() < 0.5:
            img = img[:, :, ::-1]

        # Small Gaussian noise (applied to both channels)
        noise_sigma = 0.01
        noise = self.rng.normal(loc=0.0, scale=noise_sigma, size=img.shape)
        img = img + noise.astype(img.dtype)

        # Clip flux back to [0,1], velocity to ~[-1.5,1.5] to avoid wild outliers
        img[0] = np.clip(img[0], 0.0, 1.0)
        img[1] = np.clip(img[1], -1.5, 1.5)

        return img

    def __getitem__(self, idx: int):
        # Pick a random plateifu
        pid = self.rng.choice(self.plateifus)
        src = self.source_tensors[pid]

        # Decide lensed vs unlensed
        is_lensed = self.rng.random() < self.lens_probability
        if is_lensed:
            theta_E = self.rng.uniform(*self.theta_E_range)
            img = apply_sis_lens(
                src,
                theta_E=theta_E,
                flux_mask_threshold=self.flux_mask_threshold,
            )
            label = 1
        else:
            img = src.copy()
            label = 0

        # Augment
        if self.augment:
            img = self._random_augmentation(img)

        # Convert to torch tensor
        img_t = torch.from_numpy(img).float()
        label_t = torch.tensor(label, dtype=torch.long)

        return img_t, label_t


# ---------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------

class ToyCNN(nn.Module):
    """
    Minimal 2-channel CNN for binary classification (lensed vs unlensed).

    Architecture:
      - Conv2d(in_channels, 16, 3, padding=1) + ReLU + MaxPool(2)
      - Conv2d(16, 32, 3, padding=1) + ReLU
      - AdaptiveAvgPool2d(1) -> Flatten -> Linear(32 -> 2)

    TODO (ablation): To test flux-only vs vel-only:
      - Create ToyCNN(in_channels=1) variant
      - Modify ToyLensDataset to return only channel 0 (flux) or channel 1 (vel)
      - Compare test accuracies:
          flux-only ~100% + vel-only ~50% => task is purely morphological
          both ~100% => velocity provides no additional signal for this task
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.avgpool(x)  # (B, 32, 1, 1)
        x = torch.flatten(x, 1)  # (B, 32)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Toy 2-channel CNN for lensed vs unlensed classification.")
    parser.add_argument("--tensor_dir", type=str, default="data/source_tensors",
                        help="Directory containing source_tensor_<plateifu>.npy files.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--train_samples", type=int, default=4000,
                        help="Number of synthetic samples for training dataset.")
    parser.add_argument("--val_samples", type=int, default=800,
                        help="Number of synthetic samples for validation dataset.")
    parser.add_argument("--test_samples", type=int, default=800,
                        help="Number of synthetic samples for test dataset.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output_model", type=str, default="models/toy_cnn.pt",
                        help="Path to save trained model state_dict.")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    # For this milestone we stay on CPU; change to "cuda" if you explicitly
    # want to use GPU and one is available.
    device = torch.device("cpu")

    print("============================================================")
    print("TOY CNN TRAINING (lensed vs unlensed)")
    print("============================================================")
    print(f"[INFO] tensor_dir    : {args.tensor_dir}")
    print(f"[INFO] train plateifus: {TRAIN_PLATEIFUS}")
    print(f"[INFO] val plateifus  : {VAL_PLATEIFUS}")
    print(f"[INFO] test plateifus : {TEST_PLATEIFUS}")
    print(f"[INFO] batch_size    : {args.batch_size}")
    print(f"[INFO] epochs        : {args.epochs}")
    print(f"[INFO] output_model  : {args.output_model}")
    print(f"[INFO] augment       : {not args.no_augment}")
    print("============================================================")

    # Create datasets
    train_ds = ToyLensDataset(
        plateifus=TRAIN_PLATEIFUS,
        tensor_dir=args.tensor_dir,
        n_samples=args.train_samples,
        augment=not args.no_augment,
        seed=42,
    )
    val_ds = ToyLensDataset(
        plateifus=VAL_PLATEIFUS,
        tensor_dir=args.tensor_dir,
        n_samples=args.val_samples,
        augment=False,  # keep validation deterministic-ish
        seed=43,
    )
    test_ds = ToyLensDataset(
        plateifus=TEST_PLATEIFUS,
        tensor_dir=args.tensor_dir,
        n_samples=args.test_samples,
        augment=False,
        seed=44,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # TODO: Log class balance per split (should be ~50/50 by construction)
    # Uncomment the following to verify class balance:
    # def count_classes(loader):
    #     counts = {0: 0, 1: 0}
    #     for _, labels in loader:
    #         for lbl in labels.tolist():
    #             counts[lbl] += 1
    #     return counts
    # print(f"[DEBUG] Train class balance: {count_classes(train_loader)}")
    # print(f"[DEBUG] Val class balance: {count_classes(val_loader)}")
    # print(f"[DEBUG] Test class balance: {count_classes(test_loader)}")

    # Model, loss, optimizer
    model = ToyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    # Evaluate on test set with best model (if any)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)

    print("============================================================")
    print(f"[RESULT] Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.3f}")
    print("============================================================")

    # Save best model
    torch.save(model.state_dict(), args.output_model)
    print(f"[OK] Saved model state_dict to: {args.output_model}")

    # -------------------------------------------------------------------------
    # INTERPRETATION CAVEAT
    # -------------------------------------------------------------------------
    # If test_acc ~ 100%, this confirms the pipeline is working, but the task
    # is intentionally trivial. The CNN is detecting "ring vs blob" morphology,
    # not subtle kinematic perturbations from subhalos.
    #
    # TODO: Implement baseline classifier to confirm triviality:
    #   1. Extract simple features from each image:
    #        central_flux = mean(flux[center 5x5])
    #        ring_flux = mean(flux[annulus r=20..30 pixels])
    #        ratio = ring_flux / (central_flux + 1e-6)
    #   2. Train sklearn LogisticRegression on [central_flux, ring_flux, ratio]
    #   3. If baseline acc ~ 100%, the CNN is overkill for this toy task.
    #
    # This is expected and acceptable for Milestone 1 (pipeline validation).
    # Milestone 2+ will introduce harder tasks (subhalo vs smooth).
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    main()

