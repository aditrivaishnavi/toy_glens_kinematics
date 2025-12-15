# Phase 3 - LRG Parent Catalog in Selected DR10 South Fields

In Phase 3 I construct an object level parent catalog of luminous red galaxies
(LRGs) in a small number of carefully selected high density regions within the
DESI Legacy Surveys DR10 South footprint. This catalog is the backbone for
lens simulation, machine learning training, and completeness analysis in
later phases.

## 1. Scientific goal

The goal in Phase 3 is to define and extract a realistic but manageable set of
background galaxies that live in environments where strong lensing by massive
dark matter halos is most likely. The design requirements are:

- The fields must be chosen objectively using LRG overdensities, not by
  hand picking known lenses.
- The LRG selection must be physically motivated and tied directly to the
  hyperparameter grid explored in Phase 2.
- The parent catalog must have enough objects to support robust completeness
  and bias estimates, while remaining computationally tractable for
  image based simulations and training.

From Phase 2 I learned that the **v3_color_relaxed** selection is the best
compromise between purity and completeness for this purpose, so Phase 3 adopts
v3 as the primary definition of “parent LRGs.”

## 2. Inputs

Phase 3 uses the following inputs from previous steps:

- `results/phase2_results.csv`  
  EMR Phase 2 hypergrid output with per brick LRG counts for five selection
  variants v1 through v5.

- `results/phase2_analysis/v3_color_relaxed/phase2_regions_summary.csv`  
  Summary of contiguous high density regions for the v3_color_relaxed variant.
  For each region:

  - `region_id`
  - `n_bricks` and `area_deg2`
  - `center_ra_deg`, `center_dec_deg`
  - v3 LRG surface density and percentiles

- `results/phase2_analysis/v3_color_relaxed/phase2_regions_bricks.csv`  
  Mapping from `BRICKNAME` to `region_id`, with brick level quantities:

  - `brick_ra_center`, `brick_dec_center`
  - brick area
  - Phase 2 LRG counts and densities for the five variants

- A DR10 South sweep index text file, for example
  `results/dr10/sweep_urls_full.txt`, that lists all DR10 South sweep FITS
  files used in Phases 1.5 and 2.

As in Phase 2, I restrict Phase 3 entirely to **DR10 South** (DECaLS-like
imaging). I do not include the northern MzLS/BASS imaging in this phase to
keep the instrument and systematics homogeneous.

## 3. LRG selection variants (true Phase 2 grid)

All selection variants in Phase 3 are defined to be exactly identical to those
implemented in `spark_phase2_lrg_hypergrid.py`. This is critical for internal
consistency. If I ever change the cuts in Phase 2, I must update Phase 3 and
re-run both phases.

Fluxes are in nanomaggies and are converted to AB magnitudes using the Legacy
Surveys convention

\[
m = 22.5 - 2.5 \log_{10}(\mathrm{flux}_{\mathrm{nanomaggies}}).
\]

For each object I compute

- `g_mag`, `r_mag`, `z_mag`, `w1_mag`
- `r_minus_z = r_mag - z_mag`
- `z_minus_w1 = z_mag - w1_mag`

and apply the following cuts:

- **v1_pure_massive**

  - `z_mag < 20.0`
  - `r_minus_z > 0.5`
  - `z_minus_w1 > 1.6`

  This variant selects the very brightest and reddest systems. It is aimed at
  extremely massive halos and has high purity but very low number density.

- **v2_baseline_dr10**

  - `z_mag < 20.4`
  - `r_minus_z > 0.4`
  - `z_minus_w1 > 1.6`

  This is a conservative DR10 style LRG selection. Compared to v1 it goes a
  little fainter and slightly bluer in `r_minus_z`.

- **v3_color_relaxed** (Phase 3 parent definition)

  - `z_mag < 20.4`
  - `r_minus_z > 0.4`
  - `z_minus_w1 > 0.8`

  Relative to v2, v3 keeps the same `z` limit and the same `r_minus_z` cut,
  but relaxes the `z_minus_w1` cut from 1.6 to 0.8. This admits a larger
  population of somewhat less extreme, but still massive, red galaxies.

- **v4_mag_relaxed**

  - `z_mag < 21.0`
  - `r_minus_z > 0.4`
  - `z_minus_w1 > 0.8`

  Relative to v3, v4 increases the allowed `z` magnitude, so it includes
  fainter LRGs at similar colors. This pushes towards higher completeness at
  the cost of including lower mass halos.

- **v5_very_relaxed**

  - `z_mag < 21.5`
  - `r_minus_z > 0.3`
  - `z_minus_w1 > 0.8`

  v5 is the most inclusive variant in this grid. It goes both fainter in `z`
  and slightly bluer in `r_minus_z`. The `z_minus_w1` cut remains at 0.8.
  This variant is much closer to a general luminous galaxy sample and is
  primarily useful as an upper bound on completeness rather than as a clean
  parent sample.

In the Phase 3 code I attach boolean flags `is_lrg_v1` through `is_lrg_v5`
to every candidate object and then keep only objects with `is_lrg_v3 == True`
in the parent catalog.

## 4. Region selection for Phase 3

To define the actual Phase 3 fields, I operate on the v3_color_relaxed regions
from Phase 2:

1. I load `phase2_regions_summary.csv` and compute for each region an
   area weighted score

   \[
   \mathrm{score} = \rho_{\mathrm{LRG,v3}} \times \sqrt{\mathrm{area\_deg^2}}
   \]

   where \(\rho_{\mathrm{LRG,v3}}\) is the v3 LRG surface density in that
   region. This favors regions that are both dense in LRGs and extended on
   the sky.

2. I select:

   - One primary field: the top scoring region.
   - One secondary field: the second highest scoring region, used as a
     comparison field and for robustness checks.

3. Using `phase2_regions_bricks.csv` I extract all bricks belonging to these
   selected regions and write them to

   - `results/phase3/phase3_target_bricks.csv`.

4. I summarize the choice of fields, their RA and Dec ranges, areas, and v3
   densities in

   - `results/phase3/phase3_region_choices.md`.

This process guarantees that the Phase 3 fields are chosen systematically from
the Phase 2 hypergrid results, with no hand tuning based on known lens
catalogs.

## 5. Building the parent LRG catalog

The script `scripts/run_phase3_build_parent_sample.py` performs the catalog
construction in several stages:

1. **Load target bricks and derive footprint**

   - Read `phase3_target_bricks.csv`.
   - Identify the set of Phase 3 `BRICKNAME` values and their RA and Dec
     centers.
   - Derive an RA and Dec bounding box for Phase 3 by taking the min and max
     of the brick centers and adding a small margin (0.2 degrees) on each
     side to avoid clipping edge galaxies.

2. **Loop over DR10 South sweeps**

   - Read all sweep paths or URLs from the sweep index file.
   - For each sweep, optionally use the sweep file name
     (for example `sweep-0039m320-0041m315.fits`) to quickly test whether it
     can possibly overlap the Phase 3 footprint. If it clearly cannot, skip
     it without I O.
   - For sweeps that could overlap, load the FITS table with `astropy.io.fits`
     and convert it to a Pandas DataFrame.

3. **Filter to Phase 3 bricks and footprint**

   - Restrict each sweep table to rows with:
     - `RA` and `DEC` inside the Phase 3 RA and Dec bounds, and
     - `BRICKNAME` in the Phase 3 brick list.

4. **Compute magnitudes, colors, and LRG flags**

   - Convert `FLUX_G`, `FLUX_R`, `FLUX_Z`, `FLUX_W1` to `g`, `r`, `z`,
     `w1` magnitudes.
   - Compute `r_minus_z` and `z_minus_w1`.
   - Apply the five variant cuts exactly as specified in Section 3.
   - Attach boolean flags `is_lrg_v1` through `is_lrg_v5` to each row.

5. **Keep v3 parent objects and attach region_id**

   - Keep only objects with `is_lrg_v3 == True`.
   - Map each object’s `BRICKNAME` back to its `region_id` using the bricks
     table and add this as a column.

6. **Aggregate and write output**

   - Concatenate the v3 candidates from all sweeps into a single DataFrame.
   - Write the parent catalog to

     - `results/phase3/phase3_lrg_parent_catalog.csv`

   - Optionally also write a Parquet version for faster downstream access.

The resulting parent catalog contains:

- Sky position: `RA`, `DEC`, `BRICKNAME`, `region_id`.
- Photometry: `g_mag`, `r_mag`, `z_mag`, `w1_mag`.
- Colors: `r_minus_z`, `z_minus_w1`.
- Variant membership flags: `is_lrg_v1` through `is_lrg_v5`.

This is the definitive background sample for subsequent image based phases.

## 6. Evaluation and sanity checks in Phase 3

To make sure Phase 3 is scientifically sound and internally consistent with
Phase 2, I plan the following checks:

1. **Brick level counts**

   - For each Phase 3 brick, count the number of v3 LRGs in
     `phase3_lrg_parent_catalog.csv`.
   - Compare these counts against the v3 per brick counts from
     `phase2_results.csv` restricted to the Phase 3 bricks.
   - Differences would signal issues in RA and Dec filtering, BRICKNAME
     handling, or the selection cuts.

2. **Photometric sanity**

   - Plot histograms of `z_mag`, `r_minus_z`, and `z_minus_w1` for the parent
     sample.
   - Verify that these distributions are consistent with expectations from the
     Phase 2 hypergrid analysis and with published DR10 LRG samples.

3. **Regional balance**

   - Compute the number of v3 LRGs per `region_id` in the parent catalog.
   - Check that the primary field dominates the statistics but that the
     secondary field still has enough LRGs for meaningful comparison.

4. **Scope and limitations**

   - Explicitly note in the log and later in the paper that Phase 3 covers
     only DR10 South and only a subset of the highest density v3 regions.
   - Emphasize that this is a deliberate design choice to obtain a clean,
     homogeneous training and testing environment.

These checks are necessary to support an ISEF level argument that the parent
catalog is both reliable and scientifically well motivated.

## 7. Why v3_color_relaxed is used as the parent definition

Phase 2 showed that:

- v1 and v2 are very pure but too sparse to support strong statistical
  constraints on completeness and halo level biases.
- v5 is very complete but likely contaminated by lower mass and more mixed
  galaxy populations, which dilutes the direct connection to massive halos.
- v3 strikes a useful balance for this project:

  - It significantly boosts the number of LRGs per brick relative to v2 by
    relaxing the `z_minus_w1` cut.
  - It keeps a reasonably strict `r_minus_z` cut and a conservative `z`
    limit, so it still preferentially traces massive halos, which are the
    most promising environments for strong lensing.

Because of this, Phase 3 adopts v3_color_relaxed as the parent sample
definition, while the other four variants remain attached as flags so that I
can present a full purity and completeness narrative later.

Phase 3 is the final catalog construction step before I move to actual image
cutouts and lens injection. Getting this step correct is crucial for any
later claims about the visibility of dark matter halos and the completeness of
lens searches in DR10 South.

