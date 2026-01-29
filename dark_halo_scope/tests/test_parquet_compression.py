#!/usr/bin/env python3
"""
Unit test for parquet compression migration.

Tests:
1. Create small parquet files with snappy compression
2. Run compression to gzip
3. Verify row counts match
4. Verify compression codec changed
5. Compare sizes
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def create_test_parquet(path: str, num_rows: int = 1000, compression: str = "snappy"):
    """Create a test parquet file with random data similar to Phase 4c stamps."""
    np.random.seed(42)
    
    # Simulate stamp_npz column (binary data)
    stamp_data = []
    for _ in range(num_rows):
        # Create a fake NPZ-like binary blob (64x64x3 image compressed)
        img = np.random.randn(64, 64, 3).astype(np.float32)
        import io
        bio = io.BytesIO()
        np.savez_compressed(bio, image_g=img[:,:,0], image_r=img[:,:,1], image_z=img[:,:,2])
        stamp_data.append(bio.getvalue())
    
    # Create table
    table = pa.table({
        "task_id": pa.array(range(num_rows)),
        "experiment_id": pa.array(["test_exp"] * num_rows),
        "region_split": pa.array(["train"] * (num_rows // 2) + ["val"] * (num_rows - num_rows // 2)),
        "lens_model": pa.array(["SIE"] * (num_rows // 2) + ["CONTROL"] * (num_rows - num_rows // 2)),
        "theta_e_arcsec": pa.array(np.random.uniform(0.3, 1.0, num_rows).astype(np.float64)),
        "arc_snr": pa.array(np.random.uniform(5, 50, num_rows).astype(np.float64)),
        "stamp_npz": pa.array(stamp_data, type=pa.binary()),
    })
    
    # Write with specified compression
    pq.write_table(table, path, compression=compression)
    return num_rows


def get_parquet_compression(path: str) -> str:
    """Get the compression codec used in a parquet file."""
    pf = pq.ParquetFile(path)
    # Get compression from first row group, first column chunk
    metadata = pf.metadata
    if metadata.num_row_groups > 0:
        rg = metadata.row_group(0)
        if rg.num_columns > 0:
            col = rg.column(0)
            return col.compression
    return "UNKNOWN"


def compress_parquet(source: str, target: str, compression: str = "gzip"):
    """Compress a parquet file to a new location."""
    # Read source
    table = pq.read_table(source)
    
    # Write with new compression
    pq.write_table(table, target, compression=compression)
    
    return len(table)


class TestParquetCompression(unittest.TestCase):
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp(prefix="parquet_compression_test_")
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_create_snappy_parquet(self):
        """Test creating a snappy-compressed parquet file."""
        source_path = os.path.join(self.test_dir, "source_snappy.parquet")
        num_rows = create_test_parquet(source_path, num_rows=100, compression="snappy")
        
        self.assertEqual(num_rows, 100)
        self.assertTrue(os.path.exists(source_path))
        
        # Verify compression
        compression = get_parquet_compression(source_path)
        self.assertEqual(compression.upper(), "SNAPPY")
        
        # Verify can read back
        table = pq.read_table(source_path)
        self.assertEqual(len(table), 100)
    
    def test_compress_snappy_to_gzip(self):
        """Test compressing from snappy to gzip."""
        source_path = os.path.join(self.test_dir, "source_snappy.parquet")
        target_path = os.path.join(self.test_dir, "target_gzip.parquet")
        
        # Create source
        num_rows = create_test_parquet(source_path, num_rows=500, compression="snappy")
        source_size = os.path.getsize(source_path)
        
        # Compress
        target_rows = compress_parquet(source_path, target_path, compression="gzip")
        target_size = os.path.getsize(target_path)
        
        # Verify row counts match
        self.assertEqual(num_rows, target_rows)
        
        # Verify target compression
        compression = get_parquet_compression(target_path)
        self.assertEqual(compression.upper(), "GZIP")
        
        # Verify can read back
        source_table = pq.read_table(source_path)
        target_table = pq.read_table(target_path)
        
        self.assertEqual(len(source_table), len(target_table))
        
        # Verify column values match
        for col in source_table.column_names:
            if col != "stamp_npz":  # Skip binary comparison
                self.assertTrue(
                    source_table[col].equals(target_table[col]),
                    f"Column {col} mismatch"
                )
        
        # Print size comparison
        savings_pct = (1 - target_size / source_size) * 100
        print(f"\n[TEST] Compression results:")
        print(f"  Source (snappy): {source_size:,} bytes")
        print(f"  Target (gzip):   {target_size:,} bytes")
        print(f"  Savings:         {savings_pct:.1f}%")
        
        # Gzip should be smaller (or at least not much larger)
        # For random data, compression might not help much
        self.assertGreater(source_size, 0)
        self.assertGreater(target_size, 0)
    
    def test_compress_to_zstd(self):
        """Test compressing from snappy to zstd."""
        source_path = os.path.join(self.test_dir, "source_snappy.parquet")
        target_path = os.path.join(self.test_dir, "target_zstd.parquet")
        
        # Create source
        num_rows = create_test_parquet(source_path, num_rows=500, compression="snappy")
        source_size = os.path.getsize(source_path)
        
        # Compress
        target_rows = compress_parquet(source_path, target_path, compression="zstd")
        target_size = os.path.getsize(target_path)
        
        # Verify
        self.assertEqual(num_rows, target_rows)
        compression = get_parquet_compression(target_path)
        self.assertEqual(compression.upper(), "ZSTD")
        
        print(f"\n[TEST] ZSTD compression:")
        print(f"  Source (snappy): {source_size:,} bytes")
        print(f"  Target (zstd):   {target_size:,} bytes")
        print(f"  Savings:         {(1 - target_size / source_size) * 100:.1f}%")
    
    def test_partitioned_compression(self):
        """Test compressing partitioned parquet (simplified)."""
        source_dir = os.path.join(self.test_dir, "source_partitioned")
        target_dir = os.path.join(self.test_dir, "target_partitioned")
        
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        
        # Create source files for each partition
        total_rows = 0
        for split in ["train", "val", "test"]:
            source_path = os.path.join(source_dir, f"part_{split}.parquet")
            target_path = os.path.join(target_dir, f"part_{split}.parquet")
            
            rows = create_test_parquet(source_path, num_rows=100, compression="snappy")
            total_rows += rows
            
            # Compress each file
            compress_parquet(source_path, target_path, compression="gzip")
            
            # Verify
            source_table = pq.read_table(source_path)
            target_table = pq.read_table(target_path)
            self.assertEqual(len(source_table), len(target_table))
            
            # Verify compression
            compression = get_parquet_compression(target_path)
            self.assertEqual(compression.upper(), "GZIP")
        
        print(f"\n[TEST] Partitioned compression: {total_rows} rows verified")
    
    def test_stamp_npz_integrity(self):
        """Test that stamp_npz binary data is preserved after compression."""
        source_path = os.path.join(self.test_dir, "source.parquet")
        target_path = os.path.join(self.test_dir, "target.parquet")
        
        # Create source with specific data
        num_rows = create_test_parquet(source_path, num_rows=10, compression="snappy")
        
        # Compress
        compress_parquet(source_path, target_path, compression="gzip")
        
        # Read both and compare stamp_npz
        source_table = pq.read_table(source_path)
        target_table = pq.read_table(target_path)
        
        import io
        for i in range(len(source_table)):
            source_npz = source_table["stamp_npz"][i].as_py()
            target_npz = target_table["stamp_npz"][i].as_py()
            
            # Verify bytes are identical
            self.assertEqual(source_npz, target_npz, f"stamp_npz mismatch at row {i}")
            
            # Verify can decode NPZ
            bio = io.BytesIO(target_npz)
            with np.load(bio) as npz:
                self.assertIn("image_g", npz.files)
                self.assertIn("image_r", npz.files)
                self.assertIn("image_z", npz.files)
                self.assertEqual(npz["image_g"].shape, (64, 64))
        
        print(f"\n[TEST] stamp_npz integrity verified for {num_rows} rows")


def run_size_comparison():
    """Run a size comparison test with larger dataset."""
    print("\n" + "=" * 60)
    print("Parquet Compression Size Comparison")
    print("=" * 60)
    
    test_dir = tempfile.mkdtemp(prefix="parquet_size_test_")
    
    try:
        source_path = os.path.join(test_dir, "source_snappy.parquet")
        gzip_path = os.path.join(test_dir, "target_gzip.parquet")
        zstd_path = os.path.join(test_dir, "target_zstd.parquet")
        
        # Create larger test file
        print("\nCreating test data (5000 rows with 64x64x3 stamps)...")
        num_rows = create_test_parquet(source_path, num_rows=5000, compression="snappy")
        source_size = os.path.getsize(source_path)
        
        # Compress to gzip
        print("Compressing to gzip...")
        compress_parquet(source_path, gzip_path, compression="gzip")
        gzip_size = os.path.getsize(gzip_path)
        
        # Compress to zstd
        print("Compressing to zstd...")
        compress_parquet(source_path, zstd_path, compression="zstd")
        zstd_size = os.path.getsize(zstd_path)
        
        # Results
        print("\n" + "-" * 60)
        print(f"{'Compression':<15} {'Size (bytes)':<20} {'Ratio':<10} {'Savings':<10}")
        print("-" * 60)
        print(f"{'Snappy':<15} {source_size:>15,} {'1.00x':<10} {'-':<10}")
        print(f"{'Gzip':<15} {gzip_size:>15,} {source_size/gzip_size:>6.2f}x    {(1-gzip_size/source_size)*100:>5.1f}%")
        print(f"{'Zstd':<15} {zstd_size:>15,} {source_size/zstd_size:>6.2f}x    {(1-zstd_size/source_size)*100:>5.1f}%")
        print("-" * 60)
        
        # Extrapolate to 437 GB
        train_size_gb = 437
        print(f"\nExtrapolated to {train_size_gb} GB train tier:")
        print(f"  Gzip: ~{train_size_gb * gzip_size / source_size:.0f} GB (saves ~{train_size_gb * (1 - gzip_size/source_size):.0f} GB)")
        print(f"  Zstd: ~{train_size_gb * zstd_size / source_size:.0f} GB (saves ~{train_size_gb * (1 - zstd_size/source_size):.0f} GB)")
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(exit=False, verbosity=2)
    
    # Run size comparison
    run_size_comparison()

