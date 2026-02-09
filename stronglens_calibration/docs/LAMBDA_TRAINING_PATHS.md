# Lambda Training Data Paths

**Last Updated**: 2026-02-09 08:55 UTC

## NFS Mount Point
```
/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/
```

## Directory Structure
```
stronglens_calibration/
├── cutouts/
│   ├── positives/     # 5,101 .npz files (101x101 grz cutouts)
│   └── negatives/     # 416,088 .npz files (101x101 grz cutouts)
├── manifests/         # Training manifests (to be created)
├── code/              # Training code
├── logs/              # Training logs, rclone logs
└── checkpoints/       # Model checkpoints
```

## S3 Sources
| Data | S3 Path |
|------|---------|
| Positives | `s3://darkhaloscope/stronglens_calibration/cutouts/positives/20260208_205758/` |
| Negatives | `s3://darkhaloscope/stronglens_calibration/cutouts/negatives/20260209_040454/` |
| Validation Results | `s3://darkhaloscope/stronglens_calibration/validation/20260209_085438/` |

## AWS Credentials (for rclone)
```
# Credentials stored securely - do NOT commit to git
# See: ~/.aws/credentials or contact project owner
AWS_REGION=us-east-2
```

## Rclone Remote
Already configured as `s3remote:` on Lambda instance.

## Data Stats
| Dataset | Count | Size (approx) |
|---------|-------|---------------|
| Positives | 5,101 | ~600 MB |
| Negatives | 416,088 | ~51 GB |
| **Total** | **421,189** | **~52 GB** |

## Cutout Format
- Shape: (101, 101, 3) - height, width, channels (g, r, z bands)
- Format: NPZ with keys `cutout`, `meta_*`
- Pixel scale: 0.262 arcsec/pixel

## Training Configuration
- Architecture: ResNet-18
- Input: 101x101x3
- Batch size: 128-192 (AMP enabled)
- Optimizer: AdamW with cosine LR schedule
- Epochs: 20-40 with early stopping
