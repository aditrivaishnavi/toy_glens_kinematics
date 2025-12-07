# Dark Halo Scope

## Which Dark Matter Halos Are Actually Visible?

**The Strong Lensing Selection Function of DESI Legacy DR10**

---

This project develops a quantitative, empirically calibrated **selection function** for galaxy-scale strong gravitational lenses in DESI Legacy DR10 imaging data.

### Core Question

Given real observing conditions (seeing, depth, source morphology), which dark matter halos produce *detectable* strong lensing signatures in ground-based imaging?

### Primary Observables

- **Einstein radius (θ_E)**: The angular scale of the lensing effect
- **Lens redshift (z_l)**: The distance to the lensing galaxy

### Approach

1. **Analytic theory**: Map the θ_E–z_l parameter space against DR10 seeing conditions
2. **Realistic simulations**: Inject clumpy COSMOS sources into real DR10 LRG cutouts with correct PSFs and SEDs
3. **ML detection**: Use a trained detector as a calibrated instrument to measure completeness and purity
4. **Anchor to reality**: Validate with known lenses and search for new candidates

### Key Deliverable

A selection function `C(θ_E, z_l)` that quantifies:
- Where DR10 is analytically blind (θ_E too small relative to seeing)
- Where detection is possible but uncertain
- Where reliable lens detection is expected

---

See [BLUEPRINT.md](docs/BLUEPRINT.md) for the full technical specification.

