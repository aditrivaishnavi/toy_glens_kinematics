mkdir -p external_data/dr10

cd external_data/dr10

# Download the DR10 south survey-bricks table
curl -O https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/survey-bricks-dr10-south.fits.gz

# (optional) check size
ls -lh survey-bricks-dr10-south.fits.gz

