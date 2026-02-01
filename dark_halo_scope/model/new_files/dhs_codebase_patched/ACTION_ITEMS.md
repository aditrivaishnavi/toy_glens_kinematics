# Action Items (Prioritized)

## Must-do (directly impacts FPR)
1. Train and evaluate on resolved-only regimes (min_theta_over_psf >= 0.5).
2. Use unpaired controls for a large fraction of negatives (>= 0.5).
3. Turn on PSF model = moffat for injection convolution.
4. Run stratified FPR evaluation by theta_E and theta_E/PSF.

## Should-do (improves generalization)
5. Add hard negative mining loop and retrain.
6. Enable metadata fusion using psfsize_r and psfdepth_r only (no leak columns).
7. Use curriculum training: strict resolved, higher arc_snr first, then relax.

## Medium-term (sim-to-real)
8. Replace parametric Sersic sources with real-galaxy cutouts (COSMOS or similar).
9. Replace parametric PSF with pixel-level PSF models if feasible (PSFEx or per-exposure PSF).
10. External validation on known-lens catalogs (SLACS/BELLS/others) as a sanity check.
