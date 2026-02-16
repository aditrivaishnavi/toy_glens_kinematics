Suggested Injection Models Package (Models 2 and 3)

This package is self-contained and intended to be dropped into your repo as a starting point.
It includes:
- dhs/injection_engine.py: SIE+shear injection engine (nanomaggies), PSF convolution, noise inference from psfdepth
- dhs/host_matching.py: moment-based host matching and deflector-conditioned lens parameter mapping (Model 2)
- dhs/real_sources.py: optional GalSim COSMOS source library wrapper (Model 3)
- scripts/run_selection_function.py: grid runner (supports Model 1, 2, 3)
- tests/: unit tests for critical physics invariants and host-matching logic

Requires: Python 3.11, numpy, torch, scipy, pandas, pyarrow.
Optional for Model 3: galsim + COSMOS files.
