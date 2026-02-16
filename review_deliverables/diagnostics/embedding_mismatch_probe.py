#!/usr/bin/env python3
"""Embedding mismatch probe: real lenses vs injections.

This is the fastest way to answer "does the CNN treat injections as out-of-distribution?".

It computes penultimate-layer embeddings for:
- real positives (configurable: Tier-A only, or all positives)
- injected samples generated on-the-fly (same physics engine, same preprocessing)
- negatives (optional)

Then it quantifies mismatch using:
- a simple linear classifier that tries to distinguish real positives from injections in embedding space
- MMD (kernel two-sample test) between embedding distributions
- a 2D visualization via PCA (always available) and UMAP (optional)

If a trivial classifier separates real vs injection embeddings at high accuracy (e.g., >0.8),
then your selection function based on injections is not trustworthy without correcting the mismatch.

Usage
  PYTHONPATH=. python diagnostics/embedding_mismatch_probe.py \
    --checkpoint checkpoints/.../best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --arch efficientnet_v2_s \
    --n_real 2000 --n_inj 2000 --outdir results/embedding_probe

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from dhs.model import build_model
from dhs.data import load_cutout_from_file
from dhs.preprocess import preprocess_stack
from dhs.injection_engine import inject_sie_shear
from dhs.constants import PIXEL_SCALE


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def load_model_and_hook(arch: str, checkpoint_path: str, device: str):
    model = build_model(arch, in_ch=3, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    feats: List[torch.Tensor] = []

    # Heuristic: for torchvision EfficientNetV2, the penultimate activations are model.features -> avgpool -> classifier
    # We hook the output of the avgpool (post features) by registering on model.avgpool if present;
    # fallback: hook the last features block output.
    handle = None
    if hasattr(model, "avgpool"):
        def _hook(_, __, out):
            feats.append(out)
        handle = model.avgpool.register_forward_hook(_hook)
    else:
        # generic fallback
        last = None
        for name, mod in model.named_modules():
            last = mod
        def _hook(_, __, out):
            feats.append(out)
        handle = last.register_forward_hook(_hook)

    return model, feats, handle


def featurize(model, feats_store, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    feats_store.clear()
    with torch.no_grad():
        logit = model(x).squeeze(1)
        prob = _sigmoid(logit)
    if not feats_store:
        raise RuntimeError("Feature hook did not capture activations")
    h = feats_store[-1]
    h = torch.flatten(h, start_dim=1)
    return h.cpu().numpy(), prob.cpu().numpy()


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return df.sample(n=min(n, len(df)), replace=False, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)


def load_real(df: pd.DataFrame, n: int, seed: int, tier: str | None) -> pd.DataFrame:
    d = df[df.label == 1].copy()
    if tier is not None and "tier" in d.columns:
        d = d[d.tier == tier]
    return sample_rows(d, n, seed)


def load_hosts(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    d = df[df.label == 0].copy()
    return sample_rows(d, n, seed)


@dataclass
class InjectCfg:
    theta_E_arcsec: float = 1.5
    q: float = 0.8
    phi_deg: float = 30.0
    gamma_ext: float = 0.05
    phi_ext_deg: float = 10.0
    src_mag_r: float = 23.5
    g_r: float = 0.3
    r_z: float = 0.2
    n_sersic: float = 1.0
    Re_arcsec: float = 0.25
    q_src: float = 0.7
    phi_src_deg: float = -20.0
    oversample: int = 4
    psf_fwhm_arcsec: float = 1.2


def fwhm_to_sigma_pix(fwhm_arcsec: float) -> float:
    return (fwhm_arcsec / 2.355) / PIXEL_SCALE


def inject_on_host(host_chw: np.ndarray, cfg: InjectCfg, rng: np.random.Generator) -> np.ndarray:
    host_hwc = host_chw.transpose(1, 2, 0).astype(np.float32)

    flux_r = 10 ** ((22.5 - cfg.src_mag_r) / 2.5)
    flux_g = flux_r * 10 ** (+cfg.g_r / 2.5)
    flux_z = flux_r * 10 ** (-cfg.r_z / 2.5)
    flux = np.array([flux_g, flux_r, flux_z], dtype=np.float32)

    thetaE_pix = cfg.theta_E_arcsec / PIXEL_SCALE
    phi = np.deg2rad(cfg.phi_deg)
    phi_ext = np.deg2rad(cfg.phi_ext_deg)

    beta_frac = np.sqrt(rng.uniform(0.01, 1.0))
    beta = beta_frac * thetaE_pix
    ang = rng.uniform(0, 2 * np.pi)
    bx = beta * np.cos(ang)
    by = beta * np.sin(ang)

    psf_sigma = fwhm_to_sigma_pix(cfg.psf_fwhm_arcsec)
    psf_sigmas = np.array([psf_sigma, psf_sigma, psf_sigma], dtype=np.float32)

    injected, _ = inject_sie_shear(
        host_hwc,
        thetaE_pix=float(thetaE_pix),
        q=float(cfg.q),
        phi=float(phi),
        gamma_ext=float(cfg.gamma_ext),
        phi_ext=float(phi_ext),
        x0=0.0,
        y0=0.0,
        beta_x=float(bx),
        beta_y=float(by),
        n=float(cfg.n_sersic),
        Re_pix=float(cfg.Re_arcsec / PIXEL_SCALE),
        q_src=float(cfg.q_src),
        phi_src=float(np.deg2rad(cfg.phi_src_deg)),
        flux_nmgy=flux,
        psf_sigma=psf_sigmas,
        oversample=int(cfg.oversample),
        add_clumps=False,
        rng=rng,
    )
    return injected


def prep_chw(img_hwc: np.ndarray, clip_range: float) -> np.ndarray:
    chw = img_hwc.transpose(2, 0, 1).astype(np.float32)
    x = preprocess_stack(chw, mode="raw_robust", crop=False, clip_range=clip_range)
    return x


def kernel_mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    # unbiased MMD^2 estimate with RBF kernel
    XX = np.exp(-gamma * ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    YY = np.exp(-gamma * ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    XY = np.exp(-gamma * ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    n = X.shape[0]
    m = Y.shape[0]
    # remove diagonals
    np.fill_diagonal(XX, 0.0)
    np.fill_diagonal(YY, 0.0)
    return XX.sum() / (n * (n - 1)) + YY.sum() / (m * (m - 1)) - 2.0 * XY.mean()


def fit_linear_probe(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    # simple ridge-regularized logistic regression via sklearn if available; else torch.
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        return float(accuracy_score(yte, pred))
    except Exception:
        # torch fallback
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        split = int(0.7 * len(X))
        tr, te = idx[:split], idx[split:]
        Xt = torch.tensor(X[tr], dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        Xv = torch.tensor(X[te], dtype=torch.float32)
        yv = torch.tensor(y[te], dtype=torch.float32)

        w = torch.zeros((X.shape[1], 1), dtype=torch.float32, requires_grad=True)
        b = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=1e-2)
        for _ in range(300):
            logits = Xt @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), yt)
            loss = loss + 1e-3 * (w ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = (torch.sigmoid(Xv @ w + b).squeeze(1) > 0.5).float()
            acc = (pred == yv).float().mean().item()
        return float(acc)


def pca_2d(X: np.ndarray) -> np.ndarray:
    X0 = X - X.mean(axis=0, keepdims=True)
    # SVD
    U, S, Vt = np.linalg.svd(X0, full_matrices=False)
    return X0 @ Vt[:2].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--arch", default="efficientnet_v2_s")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--split", default="val")
    ap.add_argument("--tier", default="")
    ap.add_argument("--n_real", type=int, default=2000)
    ap.add_argument("--n_inj", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--clip_range", type=float, default=10.0)
    ap.add_argument("--outdir", default="results/embedding_probe")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(args.manifest)
    # focus on requested split
    if "split" in df.columns:
        df = df[df.split == args.split].reset_index(drop=True)

    real = load_real(df, args.n_real, args.seed, tier=(args.tier if args.tier else None))
    hosts = load_hosts(df, args.n_inj, args.seed + 1)

    model, feats_store, handle = load_model_and_hook(args.arch, args.checkpoint, args.device)

    rng = np.random.default_rng(args.seed)
    icfg = InjectCfg()

    X_real = []
    P_real = []
    for p in real.cutout_path.astype(str).tolist():
        if not os.path.exists(p):
            continue
        chw = load_cutout_from_file(p)
        x = preprocess_stack(chw, mode="raw_robust", crop=False, clip_range=args.clip_range)
        xt = torch.from_numpy(x[None]).to(args.device)
        h, prob = featurize(model, feats_store, xt)
        X_real.append(h[0])
        P_real.append(prob[0])

    X_inj = []
    P_inj = []
    for p in hosts.cutout_path.astype(str).tolist():
        if not os.path.exists(p):
            continue
        host_chw = load_cutout_from_file(p)
        inj_hwc = inject_on_host(host_chw, icfg, rng)
        x = prep_chw(inj_hwc, clip_range=args.clip_range)
        xt = torch.from_numpy(x[None]).to(args.device)
        h, prob = featurize(model, feats_store, xt)
        X_inj.append(h[0])
        P_inj.append(prob[0])

    handle.remove()

    Xr = np.stack(X_real)
    Xi = np.stack(X_inj)
    Pr = np.array(P_real)
    Pi = np.array(P_inj)

    np.save(os.path.join(args.outdir, "emb_real.npy"), Xr)
    np.save(os.path.join(args.outdir, "emb_inj.npy"), Xi)
    np.save(os.path.join(args.outdir, "p_real.npy"), Pr)
    np.save(os.path.join(args.outdir, "p_inj.npy"), Pi)

    # Linear probe: can we separate real vs injection embeddings?
    X = np.concatenate([Xr, Xi], axis=0)
    y = np.concatenate([np.ones(len(Xr)), np.zeros(len(Xi))], axis=0)
    acc = fit_linear_probe(X, y, seed=args.seed)

    # MMD
    # heuristic gamma: 1 / median pairwise distance^2
    D = np.linalg.norm(X[: min(2000, len(X))][:, None, :] - X[: min(2000, len(X))][None, :, :], axis=-1)
    med = np.median(D[D > 0])
    gamma = 1.0 / (med ** 2 + 1e-12)
    mmd2 = kernel_mmd_rbf(Xr[: min(len(Xr), 2000)], Xi[: min(len(Xi), 2000)], gamma=gamma)

    # PCA plot
    Z = pca_2d(X)
    Zr = Z[: len(Xr)]
    Zi = Z[len(Xr) :]

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 6))
        plt.scatter(Zr[:, 0], Zr[:, 1], s=6, alpha=0.6, label="real")
        plt.scatter(Zi[:, 0], Zi[:, 1], s=6, alpha=0.6, label="injection")
        plt.legend()
        plt.title("Embedding mismatch probe (PCA)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "pca_real_vs_inj.png"), dpi=180)
        plt.close()
    except Exception:
        pass

    # Summary
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"n_real={len(Xr)} n_inj={len(Xi)}\n")
        f.write(f"linear_probe_acc={acc:.4f}\n")
        f.write(f"mmd2={mmd2:.6g} gamma={gamma:.3g}\n")
        f.write(f"mean_p_real={Pr.mean():.4f} mean_p_inj={Pi.mean():.4f}\n")

    print("=== Embedding mismatch probe ===")
    print(f"Saved to: {args.outdir}")
    print(f"n_real={len(Xr)}  n_inj={len(Xi)}")
    print(f"Linear probe accuracy (real vs injection): {acc:.4f}")
    print(f"MMD^2 (RBF) between distributions: {mmd2:.6g}")
    print(f"Mean model score: real={Pr.mean():.4f}  injection={Pi.mean():.4f}")


if __name__ == "__main__":
    main()
