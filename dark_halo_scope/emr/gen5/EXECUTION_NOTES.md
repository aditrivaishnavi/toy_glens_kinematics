# Gen5 Execution Notes

## Current Status: Ready for emr-launcher Execution

### Completed Locally
- ✅ Directory structure created (configs/gen5, emr/gen5, models/gen5_cosmos, data/gen5)
- ✅ cosmos_source_loader_v2.py created with all fixes:
  - Float32 precision (not float16)
  - HLR filtering (0.1-1.0 arcsec)
  - Config file support
  - Checkpointing (saves every 1000 sources)
- ✅ cosmos_bank_config.json created

### Next Steps (Run on emr-launcher)

**IMPORTANT**: The remaining steps MUST be executed on emr-launcher, not locally.

1. **COSMOS Bank Building** (cosmos-bank-build todo)
   - SSH to emr-launcher
   - Run: `python -m src.sims.cosmos_source_loader_v2 --config configs/gen5/cosmos_bank_config.json --verbose`
   - If interrupted, rerun same command (will resume from checkpoint)
   - Upload result to S3

2. **Spark Pipeline Integration** (multiple todos)
   - Copy `emr/spark_phase4_pipeline.py` to `emr/gen5/spark_phase4_pipeline_gen5.py`
   - Add COSMOS helper functions
   - Add render_cosmos_lensed_source() function
   - Add config support
   - Modify Stage 4c injection logic
   - Upload to S3

3. **EMR Execution**
   - Create bootstrap script
   - Launch cluster from emr-launcher
   - Submit Phase 4c job

### Why emr-launcher?
- Reliable connectivity (no laptop network issues)
- Access to AWS credentials
- Can monitor long-running jobs
- Same environment as EMR cluster
DOC

EOF
echo "Documentation created"