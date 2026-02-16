from __future__ import annotations
import os, argparse, time, json
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from training._common import seed_all, make_loaders, effective_accum_steps, save_preds_parquet
from stronglens.metrics import safe_auc, pr_at_thresh
from stronglens.train_utils import weighted_bce_loss, save_run_info, RunInfo, predict_logits

def make_step_halve_lr(optimizer, step_epoch: int):
    def lr_lambda(epoch: int):
        return 0.5 if epoch >= step_epoch else 1.0
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--base-lr", type=float, default=3.88e-4)
    ap.add_argument("--lr-step-epoch", type=int, default=130)
    ap.add_argument("--effective-batch", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--crop-size", type=int, default=0, help="0 means no crop (101x101). Set 64 to reproduce your earlier crop.")
    ap.add_argument("--pretrained", action="store_true", default=True)
    args = ap.parse_args()

    crop_size = None if args.crop_size == 0 else int(args.crop_size)

    os.makedirs(args.outdir, exist_ok=True)
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train, ds_val, train_loader, val_loader = make_loaders(args.manifest, args.micro_batch, args.num_workers, crop_size)
    accum = effective_accum_steps(args.effective_batch, args.micro_batch)

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if args.pretrained else None
    model = efficientnet_b0(weights=weights)
    # replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.base_lr)
    scheduler = make_step_halve_lr(optimizer, args.lr_step_epoch)

    best_auc = -1.0
    best_path = os.path.join(args.outdir, "best.pt")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        nsteps = 0

        for i, batch in enumerate(train_loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            w = batch["w"].to(device, non_blocking=True)

            logits = model(x)
            loss = weighted_bce_loss(logits, y, w) / accum
            loss.backward()
            running += float(loss.item())
            nsteps += 1

            if (i + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        y_true, val_logits, tiers = predict_logits(model, val_loader, device)
        scores = 1/(1+np.exp(-np.asarray(val_logits)))
        auc = safe_auc(np.asarray(y_true), scores)
        pr = pr_at_thresh(np.asarray(y_true), scores, 0.5)

        row = {"epoch": epoch+1, "train_loss_scaled": running/max(nsteps,1), "val_auc": auc, **{f"val_{k}@0.5":v for k,v in pr.items()}, "lr": optimizer.param_groups[0]["lr"]}
        with open(os.path.join(args.outdir, "history.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")

        if auc == auc and auc > best_auc:
            best_auc = auc
            torch.save({"model": model.state_dict(), "epoch": epoch+1, "val_auc": auc, "args": vars(args)}, best_path)

        print(f"Epoch {epoch+1:03d}/{args.epochs} loss={row['train_loss_scaled']:.4g} val_auc={auc:.4f} lr={row['lr']:.3g}")

    y_true, val_logits, tiers = predict_logits(model, val_loader, device)
    save_preds_parquet(os.path.join(args.outdir, "preds_val.parquet"), y_true, val_logits, tiers)

    save_run_info(args.outdir, RunInfo(command=" ".join(os.sys.argv), config=vars(args)))

if __name__ == "__main__":
    main()
