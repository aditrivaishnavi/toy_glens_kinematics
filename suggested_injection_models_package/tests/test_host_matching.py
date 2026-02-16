
import unittest
import numpy as np
from dhs.host_matching import estimate_host_moments_rband

class TestHostMoments(unittest.TestCase):
    def test_round_gaussian(self):
        H=W=101
        yy,xx=np.mgrid[0:H,0:W]
        cy=(H-1)/2; cx=(W-1)/2
        sig=8.0
        img=np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/sig**2).astype(np.float32)
        host=np.zeros((H,W,3),dtype=np.float32)
        host[...,1]=img
        m=estimate_host_moments_rband(host)
        self.assertTrue(0.9 <= m.q <= 1.0)

    def test_elliptical(self):
        H=W=101
        yy,xx=np.mgrid[0:H,0:W]
        cy=(H-1)/2; cx=(W-1)/2
        sigx=10.0; sigy=5.0
        img=np.exp(-0.5*(((xx-cx)/sigx)**2+((yy-cy)/sigy)**2)).astype(np.float32)
        host=np.zeros((H,W,3),dtype=np.float32)
        host[...,1]=img
        m=estimate_host_moments_rband(host)
        self.assertTrue(m.q < 0.8)

if __name__ == "__main__":
    unittest.main()
