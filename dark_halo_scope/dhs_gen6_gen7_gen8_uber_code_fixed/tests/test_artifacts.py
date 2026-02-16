import numpy as np
from dhs_gen.domain_randomization.artifacts import apply_domain_randomization, ArtifactConfig

def test_artifacts_runs():
    img = np.zeros((64,64), dtype=np.float32)
    img[32,32] = 1.0
    out = apply_domain_randomization(img, key="task_1", psf_fwhm_pix=3.0, cfg=ArtifactConfig())
    assert out["img"].shape == img.shape
    assert np.isfinite(out["img"]).all()
