#!/usr/bin/env python3
"""
Data caching abstraction for Phase 5 training.

Provides transparent caching of S3 data to local storage.
Training code can request data by S3 path, and this module will:
1. Check if data exists in local cache
2. Download from S3 if not cached
3. Return local path for training

Usage:
    from data_cache import DataCache
    
    cache = DataCache(cache_root="/data/cache")
    local_path = cache.get("s3://darkhaloscope/phase4_pipeline/phase4c/.../stamps/train_stamp64_bandsgrz_gridgrid_small")
    # Returns: /data/cache/phase4c/stamps/train_stamp64_bandsgrz_gridgrid_small
    
    # Training code uses local_path directly
    parquet_files = glob.glob(f"{local_path}/**/*.parquet", recursive=True)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import glob


@dataclass
class CacheEntry:
    """Represents a cached dataset."""
    s3_uri: str
    local_path: str
    file_count: int
    size_bytes: int
    cached_at: str
    verified: bool = False


@dataclass
class CacheManifest:
    """Tracks all cached datasets."""
    entries: Dict[str, CacheEntry] = field(default_factory=dict)
    
    def save(self, path: str):
        data = {
            k: {
                "s3_uri": v.s3_uri,
                "local_path": v.local_path,
                "file_count": v.file_count,
                "size_bytes": v.size_bytes,
                "cached_at": v.cached_at,
                "verified": v.verified,
            }
            for k, v in self.entries.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "CacheManifest":
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            data = json.load(f)
        entries = {
            k: CacheEntry(**v) for k, v in data.items()
        }
        return cls(entries=entries)


class DataCache:
    """
    Transparent caching layer for S3 data.
    
    Args:
        cache_root: Local directory for cached data (e.g., /data/cache)
        manifest_file: Path to cache manifest JSON (default: {cache_root}/manifest.json)
    """
    
    def __init__(self, cache_root: str = "/data/cache", manifest_file: Optional[str] = None):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        self.manifest_file = manifest_file or str(self.cache_root / "manifest.json")
        self.manifest = CacheManifest.load(self.manifest_file)
    
    def _s3_to_local_path(self, s3_uri: str) -> str:
        """Convert S3 URI to local cache path."""
        # s3://darkhaloscope/phase4_pipeline/phase4c/.../stamps/experiment_id
        # -> /data/cache/darkhaloscope/phase4_pipeline/phase4c/.../stamps/experiment_id
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got: {s3_uri}")
        
        # Remove s3:// prefix
        path_part = s3_uri[5:]
        return str(self.cache_root / path_part)
    
    def _cache_key(self, s3_uri: str) -> str:
        """Generate cache key from S3 URI."""
        return hashlib.md5(s3_uri.encode()).hexdigest()[:16]
    
    def _count_parquet_files(self, local_path: str) -> int:
        """Count parquet files in a directory."""
        files = glob.glob(os.path.join(local_path, "**", "*.parquet"), recursive=True)
        return len(files)
    
    def _get_dir_size(self, local_path: str) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for dirpath, dirnames, filenames in os.walk(local_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total
    
    def _sync_from_s3(self, s3_uri: str, local_path: str, quiet: bool = False) -> bool:
        """Download data from S3 to local path using aws s3 sync."""
        os.makedirs(local_path, exist_ok=True)
        
        cmd = ["aws", "s3", "sync", s3_uri, local_path]
        if quiet:
            cmd.append("--quiet")
        
        print(f"[DataCache] Syncing from S3...")
        print(f"  Source: {s3_uri}")
        print(f"  Destination: {local_path}")
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=not quiet)
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"[DataCache] ERROR: aws s3 sync failed with code {result.returncode}")
            if result.stderr:
                print(result.stderr.decode())
            return False
        
        print(f"[DataCache] Sync completed in {elapsed:.1f}s")
        return True
    
    def is_cached(self, s3_uri: str) -> bool:
        """Check if data is already cached locally."""
        key = self._cache_key(s3_uri)
        if key not in self.manifest.entries:
            return False
        
        entry = self.manifest.entries[key]
        local_path = entry.local_path
        
        # Verify local path exists and has files
        if not os.path.exists(local_path):
            return False
        
        current_count = self._count_parquet_files(local_path)
        if current_count == 0:
            return False
        
        # Allow some tolerance (S3 might have a few more files)
        if current_count < entry.file_count * 0.95:
            print(f"[DataCache] Warning: Cache incomplete. Expected {entry.file_count} files, found {current_count}")
            return False
        
        return True
    
    def get(self, s3_uri: str, force_refresh: bool = False) -> str:
        """
        Get local path for data, downloading from S3 if necessary.
        
        Args:
            s3_uri: S3 URI (e.g., s3://bucket/path/to/data)
            force_refresh: If True, re-download even if cached
            
        Returns:
            Local path to cached data
        """
        s3_uri = s3_uri.rstrip("/")
        key = self._cache_key(s3_uri)
        local_path = self._s3_to_local_path(s3_uri)
        
        # Check cache
        if not force_refresh and self.is_cached(s3_uri):
            entry = self.manifest.entries[key]
            print(f"[DataCache] Using cached data:")
            print(f"  S3: {s3_uri}")
            print(f"  Local: {local_path}")
            print(f"  Files: {entry.file_count}, Size: {entry.size_bytes / 1e9:.1f} GB")
            print(f"  Cached at: {entry.cached_at}")
            return local_path
        
        # Download from S3
        print(f"[DataCache] Data not cached, downloading from S3...")
        success = self._sync_from_s3(s3_uri, local_path)
        
        if not success:
            raise RuntimeError(f"Failed to sync data from {s3_uri}")
        
        # Update manifest
        file_count = self._count_parquet_files(local_path)
        size_bytes = self._get_dir_size(local_path)
        
        entry = CacheEntry(
            s3_uri=s3_uri,
            local_path=local_path,
            file_count=file_count,
            size_bytes=size_bytes,
            cached_at=datetime.now().isoformat(),
            verified=True,
        )
        self.manifest.entries[key] = entry
        self.manifest.save(self.manifest_file)
        
        print(f"[DataCache] Cached {file_count} files ({size_bytes / 1e9:.1f} GB)")
        return local_path
    
    def get_or_local(self, path: str, force_refresh: bool = False) -> str:
        """
        Smart path resolution: if S3 URI, cache and return local path.
        If already local path, return as-is.
        
        Args:
            path: Either S3 URI or local path
            force_refresh: If True and S3 URI, re-download
            
        Returns:
            Local path to data
        """
        if path.startswith("s3://"):
            return self.get(path, force_refresh=force_refresh)
        else:
            # Already a local path
            if not os.path.exists(path):
                raise FileNotFoundError(f"Local path does not exist: {path}")
            return path
    
    def list_cached(self) -> List[CacheEntry]:
        """List all cached datasets."""
        return list(self.manifest.entries.values())
    
    def clear(self, s3_uri: Optional[str] = None):
        """
        Clear cache. If s3_uri is provided, clear only that entry.
        Otherwise, clear all cached data.
        """
        import shutil
        
        if s3_uri:
            key = self._cache_key(s3_uri)
            if key in self.manifest.entries:
                entry = self.manifest.entries[key]
                if os.path.exists(entry.local_path):
                    shutil.rmtree(entry.local_path)
                    print(f"[DataCache] Cleared: {entry.local_path}")
                del self.manifest.entries[key]
                self.manifest.save(self.manifest_file)
        else:
            for entry in self.manifest.entries.values():
                if os.path.exists(entry.local_path):
                    shutil.rmtree(entry.local_path)
                    print(f"[DataCache] Cleared: {entry.local_path}")
            self.manifest.entries.clear()
            self.manifest.save(self.manifest_file)
    
    def status(self) -> Dict:
        """Get cache status summary."""
        total_files = 0
        total_bytes = 0
        for entry in self.manifest.entries.values():
            total_files += entry.file_count
            total_bytes += entry.size_bytes
        
        return {
            "cache_root": str(self.cache_root),
            "num_datasets": len(self.manifest.entries),
            "total_files": total_files,
            "total_size_gb": total_bytes / 1e9,
            "datasets": [
                {
                    "s3_uri": e.s3_uri,
                    "local_path": e.local_path,
                    "file_count": e.file_count,
                    "size_gb": e.size_bytes / 1e9,
                    "cached_at": e.cached_at,
                }
                for e in self.manifest.entries.values()
            ]
        }


# Convenience function for scripts
def get_cached_path(s3_or_local_path: str, cache_root: str = "/data/cache") -> str:
    """
    Convenience function: get local path, caching from S3 if needed.
    
    Args:
        s3_or_local_path: Either S3 URI or local path
        cache_root: Local cache directory
        
    Returns:
        Local path to data
    """
    cache = DataCache(cache_root=cache_root)
    return cache.get_or_local(s3_or_local_path)


if __name__ == "__main__":
    # CLI for cache management
    import argparse
    
    parser = argparse.ArgumentParser(description="Data cache management")
    parser.add_argument("--cache-root", default="/data/cache", help="Cache directory")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # status command
    subparsers.add_parser("status", help="Show cache status")
    
    # get command
    get_parser = subparsers.add_parser("get", help="Get/cache data from S3")
    get_parser.add_argument("s3_uri", help="S3 URI to cache")
    get_parser.add_argument("--force", action="store_true", help="Force re-download")
    
    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument("--s3-uri", help="Clear specific S3 URI (default: clear all)")
    
    args = parser.parse_args()
    
    cache = DataCache(cache_root=args.cache_root)
    
    if args.command == "status":
        status = cache.status()
        print(json.dumps(status, indent=2))
    
    elif args.command == "get":
        local_path = cache.get(args.s3_uri, force_refresh=args.force)
        print(f"Local path: {local_path}")
    
    elif args.command == "clear":
        cache.clear(s3_uri=args.s3_uri)
        print("Cache cleared.")
    
    else:
        parser.print_help()

