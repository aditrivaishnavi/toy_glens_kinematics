import numpy as np
from dhs_gen.hybrid_sources.sersic_clumps import generate_hybrid_source

def test_hybrid_deterministic():
    a = generate_hybrid_source("task_123", salt="x")["img"]
    b = generate_hybrid_source("task_123", salt="x")["img"]
    assert np.allclose(a, b)
    c = generate_hybrid_source("task_123", salt="y")["img"]
    assert not np.allclose(a, c)
    assert abs(float(a.sum()) - 1.0) < 1e-6
