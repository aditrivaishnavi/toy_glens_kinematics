import io
import gzip
import numpy as np

from dhs_gen.utils import bilinear_resample
from dhs_gen.validation.quality_checks import decode_stamp_npz


def test_bilinear_resample_flux_conservation():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(64, 64)).astype(np.float32)
    img -= img.min()
    s0 = float(np.sum(img))

    out = bilinear_resample(img, scale_y=0.5, scale_x=0.5)
    s1 = float(np.sum(out))
    # Allow small interpolation error
    assert abs(s0 - s1) / (s0 + 1e-6) < 0.02


def test_decode_stamp_npz_multiband_keys():
    rng = np.random.default_rng(1)
    g = rng.normal(size=(64, 64)).astype(np.float32)
    r = rng.normal(size=(64, 64)).astype(np.float32)
    z = rng.normal(size=(64, 64)).astype(np.float32)

    buf = io.BytesIO()
    np.savez_compressed(buf, image_g=g, image_r=r, image_z=z)
    blob = buf.getvalue()

    arr, bandset = decode_stamp_npz(blob)
    assert bandset == "grz"
    assert arr.shape == (3, 64, 64)

    # gzip-wrapped should also work
    gz = gzip.compress(blob)
    arr2, bandset2 = decode_stamp_npz(gz)
    assert bandset2 == "grz"
    assert np.allclose(arr, arr2)
