"""
Environment check script for the toy galaxy lensing kinematics project.

Verifies that all required dependencies are installed and prints version info.
"""

import numpy as np
import torch
import astropy
import matplotlib


def check_environment():
    """Print versions of key packages and check CUDA availability."""
    print("=== Environment Check ===\n")

    print(f"NumPy version:      {np.__version__}")
    print(f"PyTorch version:    {torch.__version__}")
    print(f"Astropy version:    {astropy.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")

    print()
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available:     {cuda_available}")

    if cuda_available:
        print(f"CUDA version:       {torch.version.cuda}")
        print(f"GPU device:         {torch.cuda.get_device_name(0)}")
    else:
        print("(Running on CPU only)")

    print("\n=== All checks passed ===")


if __name__ == "__main__":
    check_environment()

