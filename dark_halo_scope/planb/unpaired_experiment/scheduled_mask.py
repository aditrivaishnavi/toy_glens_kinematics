
"""
scheduled_mask.py

Production-ready scheduled core masking for lens-finding training.

Masking operates on torch.Tensor images with shape:
- (C,H,W) or (H,W) or batched (N,C,H,W) / (N,H,W).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import torch


@dataclass(frozen=True)
class ScheduleEntry:
    """One schedule segment starting at epoch_start (inclusive)."""
    epoch_start: int
    radius: int
    prob: float


class ScheduledCoreMask:
    """
    Scheduled, radially symmetric core masking.

    schedule_entries: list of ScheduleEntry or tuples (epoch_start, radius, prob)
      - Entries are applied in increasing epoch_start order.
      - For a given epoch, the last entry with epoch_start <= epoch is used.

    image_size:
      - H == W expected; if not, the min(H,W) is used to compute center/mask.
    fill_value:
      - Value to fill masked pixels with (0.0 recommended if images are normalized).
    soft_edge_sigma:
      - If provided, uses a soft (Gaussian) edge instead of a hard disk.
      - sigma is in pixels; recommended 0.75 to 1.5 if you want soft edges.
    """

    def __init__(
        self,
        schedule_entries: Union[List[ScheduleEntry], List[Tuple[int, int, float]]],
        image_size: int = 64,
        fill_value: float = 0.0,
        soft_edge_sigma: Optional[float] = None,
    ) -> None:
        if len(schedule_entries) == 0:
            raise ValueError("schedule_entries must be non-empty")

        entries: List[ScheduleEntry] = []
        for e in schedule_entries:
            if isinstance(e, ScheduleEntry):
                entries.append(e)
            else:
                epoch_start, radius, prob = e
                entries.append(ScheduleEntry(int(epoch_start), int(radius), float(prob)))

        entries = sorted(entries, key=lambda x: x.epoch_start)

        if entries[0].epoch_start != 0:
            raise ValueError("First schedule entry must start at epoch 0")

        for i, ent in enumerate(entries):
            if ent.radius < 0:
                raise ValueError(f"radius must be >= 0, got {ent.radius}")
            if not (0.0 <= ent.prob <= 1.0):
                raise ValueError(f"prob must be in [0,1], got {ent.prob}")
            if i > 0 and ent.epoch_start == entries[i - 1].epoch_start:
                raise ValueError("Duplicate epoch_start values are not allowed")

        self._entries = entries
        self.image_size = int(image_size)
        self.fill_value = float(fill_value)
        self.soft_edge_sigma = soft_edge_sigma

        # Cache: {(device, dtype, H, W, radius, soft_sigma): mask_tensor}
        self._mask_cache = {}

    def get_current_params(self, epoch: int) -> Tuple[int, float]:
        ent = self._entry_for_epoch(epoch)
        return ent.radius, ent.prob

    def __call__(
        self,
        img: torch.Tensor,
        epoch: int,
        *,
        deterministic: bool = False,
        force_apply: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(img):
            raise TypeError("img must be a torch.Tensor")

        ent = self._entry_for_epoch(epoch)
        radius, prob = ent.radius, ent.prob

        if radius <= 0 or prob <= 0.0:
            return img

        if force_apply is True:
            apply = True
        elif force_apply is False:
            apply = False
        else:
            if deterministic:
                apply = True
            else:
                r = torch.rand((), device=img.device, generator=generator)
                apply = bool(r.item() < prob)

        if not apply:
            return img

        return apply_deterministic_mask(
            img,
            radius=radius,
            fill_value=self.fill_value,
            soft_edge_sigma=self.soft_edge_sigma,
            _cache=self._mask_cache,
        )

    def _entry_for_epoch(self, epoch: int) -> ScheduleEntry:
        e = int(epoch)
        current = self._entries[0]
        for ent in self._entries:
            if ent.epoch_start <= e:
                current = ent
            else:
                break
        return current


def apply_deterministic_mask(
    img: torch.Tensor,
    radius: int,
    *,
    fill_value: float = 0.0,
    soft_edge_sigma: Optional[float] = None,
    _cache: Optional[dict] = None,
) -> torch.Tensor:
    """
    Apply a deterministic radially symmetric mask of given radius.

    Supports shapes:
      - (H,W)
      - (C,H,W)
      - (N,H,W)
      - (N,C,H,W)
    """
    if radius <= 0:
        return img

    original_shape = img.shape
    x = img

    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        mode = "HW"
    elif x.dim() == 3:
        # For 3D, assume CHW if first dim is <= 6 (number of channels)
        # This is a heuristic - 3-channel RGB/grz is most common
        # For batches of 1-6 2D images, user should unsqueeze to 4D
        if original_shape[0] <= 6:
            x = x.unsqueeze(0)
            mode = "CHW"
        else:
            x = x.unsqueeze(1)
            mode = "NHW"
    elif x.dim() == 4:
        mode = "NCHW"
    else:
        raise ValueError(f"Unsupported img dim {x.dim()}")

    N, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    key = None
    mask = None
    if _cache is not None:
        key = (str(device), str(dtype), H, W, int(radius), None if soft_edge_sigma is None else float(soft_edge_sigma))
        mask = _cache.get(key)

    if mask is None:
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
        xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
        rr = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        if soft_edge_sigma is None:
            mask2d = (rr >= float(radius)).to(dtype)
        else:
            sigma = float(soft_edge_sigma)
            t = (float(radius) - rr) / max(sigma, 1e-6)
            w = torch.exp(-0.5 * torch.clamp(t, min=0.0) ** 2)
            mask2d = (1.0 - w).to(dtype)

        mask = mask2d.view(1, 1, H, W)
        if _cache is not None and key is not None:
            _cache[key] = mask

    if fill_value == 0.0:
        out = x * mask
    else:
        fv = torch.tensor(float(fill_value), device=device, dtype=dtype)
        out = x * mask + fv * (1.0 - mask)

    if mode == "HW":
        return out[0, 0]
    if mode == "CHW":
        return out[0]
    if mode == "NHW":
        return out[:, 0]
    return out
