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
from typing import Dict, List, Optional, Tuple
import glob


@dataclass
class CacheEntry:
    """Represents a cached dataset."""
    s3_uri: str
    local_path: str
    file_count: int
    size_bytes: int
    cached_at: str
    last_accessed: str = ""
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
    Transparent caching layer for S3 data with disk safety.
    
    Args:
        cache_root: Local directory for cached data (e.g., /data/cache)
        manifest_file: Path to cache manifest JSON (default: {cache_root}/manifest.json)
        max_cache_gb: Maximum cache size in GB (default: 900 for 1TB disk)
        min_free_gb: Minimum free disk space to maintain (default: 50 GB)
        warn_threshold: Warn when disk usage exceeds this fraction (default: 0.8)
    """
    
    # Default limits for 1TB disk
    DEFAULT_MAX_CACHE_GB = 900  # Leave 100GB headroom
    DEFAULT_MIN_FREE_GB = 50   # Always keep 50GB free
    
    def __init__(
        self, 
        cache_root: str = "/data/cache", 
        manifest_file: Optional[str] = None,
        max_cache_gb: float = DEFAULT_MAX_CACHE_GB,
        min_free_gb: float = DEFAULT_MIN_FREE_GB,
        warn_threshold: float = 0.8,
    ):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        self.manifest_file = manifest_file or str(self.cache_root / "manifest.json")
        self.manifest = CacheManifest.load(self.manifest_file)
        
        # Disk safety limits
        self.max_cache_bytes = int(max_cache_gb * 1e9)
        self.min_free_bytes = int(min_free_gb * 1e9)
        self.warn_threshold = warn_threshold
    
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
    
    def _get_disk_usage(self) -> Dict[str, int]:
        """Get disk usage statistics for cache root."""
        import shutil
        total, used, free = shutil.disk_usage(self.cache_root)
        return {"total": total, "used": used, "free": free}
    
    def _get_cache_size(self) -> int:
        """Get total size of all cached data."""
        return sum(e.size_bytes for e in self.manifest.entries.values())
    
    def _check_disk_space(self, required_bytes: int) -> Tuple[bool, str]:
        """
        Check if we have enough disk space for a download.
        
        Returns:
            (ok, message) - ok is True if safe to proceed
        """
        disk = self._get_disk_usage()
        cache_size = self._get_cache_size()
        
        # Check 1: Would exceed max cache size?
        if cache_size + required_bytes > self.max_cache_bytes:
            return False, (
                f"Download would exceed max cache size.\n"
                f"  Current cache: {cache_size / 1e9:.1f} GB\n"
                f"  Required: {required_bytes / 1e9:.1f} GB\n"
                f"  Max allowed: {self.max_cache_bytes / 1e9:.1f} GB\n"
                f"  Suggestion: Clear old cache with `python data_cache.py clear`"
            )
        
        # Check 2: Would leave too little free space?
        if disk["free"] - required_bytes < self.min_free_bytes:
            return False, (
                f"Download would leave insufficient free space.\n"
                f"  Current free: {disk['free'] / 1e9:.1f} GB\n"
                f"  Required: {required_bytes / 1e9:.1f} GB\n"
                f"  Min free required: {self.min_free_bytes / 1e9:.1f} GB\n"
                f"  Suggestion: Clear old cache or increase disk size"
            )
        
        # Check 3: Warn if approaching threshold
        usage_after = (disk["used"] + required_bytes) / disk["total"]
        if usage_after > self.warn_threshold:
            print(f"[DataCache] WARNING: Disk usage will be {usage_after*100:.1f}% after download")
        
        return True, "OK"
    
    def _estimate_s3_size(self, s3_uri: str) -> int:
        """Estimate size of S3 dataset (uses aws s3 ls --summarize)."""
        import subprocess
        
        try:
            cmd = ["aws", "s3", "ls", "--recursive", "--summarize", s3_uri]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"[DataCache] Warning: Could not estimate S3 size, using default estimate")
                return 500 * 1024 * 1024 * 1024  # 500 GB default
            
            # Parse "Total Size: 123456789" from output
            for line in result.stdout.split("\n"):
                if "Total Size:" in line:
                    size_str = line.split(":")[-1].strip()
                    return int(size_str)
            
            return 500 * 1024 * 1024 * 1024  # 500 GB default
        except Exception as e:
            print(f"[DataCache] Warning: Error estimating S3 size: {e}")
            return 500 * 1024 * 1024 * 1024  # 500 GB default
    
    def _evict_lru(self, required_bytes: int) -> bool:
        """
        Evict least-recently-used cache entries to free space.
        
        Returns True if enough space was freed.
        """
        import shutil
        
        if not self.manifest.entries:
            return False
        
        # Sort by last_accessed (oldest first)
        entries = sorted(
            self.manifest.entries.items(),
            key=lambda x: x[1].last_accessed or x[1].cached_at
        )
        
        freed = 0
        for key, entry in entries:
            if freed >= required_bytes:
                break
            
            print(f"[DataCache] Evicting LRU entry: {entry.s3_uri[:60]}...")
            if os.path.exists(entry.local_path):
                shutil.rmtree(entry.local_path)
                freed += entry.size_bytes
            del self.manifest.entries[key]
        
        self.manifest.save(self.manifest_file)
        print(f"[DataCache] Freed {freed / 1e9:.1f} GB")
        return freed >= required_bytes
    
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
    
    def get(self, s3_uri: str, force_refresh: bool = False, auto_evict: bool = True) -> str:
        """
        Get local path for data, downloading from S3 if necessary.
        
        Args:
            s3_uri: S3 URI (e.g., s3://bucket/path/to/data)
            force_refresh: If True, re-download even if cached
            auto_evict: If True, automatically evict LRU entries if disk full
            
        Returns:
            Local path to cached data
            
        Raises:
            RuntimeError: If disk is full and cannot evict enough space
        """
        s3_uri = s3_uri.rstrip("/")
        key = self._cache_key(s3_uri)
        local_path = self._s3_to_local_path(s3_uri)
        
        # Check cache
        if not force_refresh and self.is_cached(s3_uri):
            entry = self.manifest.entries[key]
            # Update last_accessed
            entry.last_accessed = datetime.now().isoformat()
            self.manifest.save(self.manifest_file)
            
            print(f"[DataCache] Using cached data:")
            print(f"  S3: {s3_uri}")
            print(f"  Local: {local_path}")
            print(f"  Files: {entry.file_count}, Size: {entry.size_bytes / 1e9:.1f} GB")
            print(f"  Cached at: {entry.cached_at}")
            return local_path
        
        # Estimate download size
        print(f"[DataCache] Estimating S3 dataset size...")
        estimated_size = self._estimate_s3_size(s3_uri)
        print(f"[DataCache] Estimated size: {estimated_size / 1e9:.1f} GB")
        
        # Check disk space
        ok, message = self._check_disk_space(estimated_size)
        if not ok:
            if auto_evict:
                print(f"[DataCache] Disk space insufficient, attempting LRU eviction...")
                evicted = self._evict_lru(estimated_size)
                if evicted:
                    ok, message = self._check_disk_space(estimated_size)
            
            if not ok:
                raise RuntimeError(f"[DataCache] DISK FULL - Cannot download.\n{message}")
        
        # Show disk status before download
        disk = self._get_disk_usage()
        print(f"[DataCache] Disk status before download:")
        print(f"  Total: {disk['total'] / 1e9:.1f} GB")
        print(f"  Used: {disk['used'] / 1e9:.1f} GB ({disk['used']/disk['total']*100:.1f}%)")
        print(f"  Free: {disk['free'] / 1e9:.1f} GB")
        print(f"  Cache size: {self._get_cache_size() / 1e9:.1f} GB")
        
        # Download from S3
        print(f"[DataCache] Downloading from S3...")
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
            last_accessed=datetime.now().isoformat(),
            verified=True,
        )
        self.manifest.entries[key] = entry
        self.manifest.save(self.manifest_file)
        
        # Show disk status after download
        disk = self._get_disk_usage()
        print(f"[DataCache] Cached {file_count} files ({size_bytes / 1e9:.1f} GB)")
        print(f"[DataCache] Disk after download: {disk['free'] / 1e9:.1f} GB free ({disk['used']/disk['total']*100:.1f}% used)")
        
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
        """Get cache status summary with disk usage."""
        total_files = 0
        total_bytes = 0
        for entry in self.manifest.entries.values():
            total_files += entry.file_count
            total_bytes += entry.size_bytes
        
        disk = self._get_disk_usage()
        
        return {
            "cache_root": str(self.cache_root),
            "disk": {
                "total_gb": disk["total"] / 1e9,
                "used_gb": disk["used"] / 1e9,
                "free_gb": disk["free"] / 1e9,
                "used_pct": disk["used"] / disk["total"] * 100,
            },
            "limits": {
                "max_cache_gb": self.max_cache_bytes / 1e9,
                "min_free_gb": self.min_free_bytes / 1e9,
                "warn_threshold_pct": self.warn_threshold * 100,
            },
            "cache": {
                "num_datasets": len(self.manifest.entries),
                "total_files": total_files,
                "total_size_gb": total_bytes / 1e9,
                "headroom_gb": (self.max_cache_bytes - total_bytes) / 1e9,
            },
            "datasets": [
                {
                    "s3_uri": e.s3_uri,
                    "local_path": e.local_path,
                    "file_count": e.file_count,
                    "size_gb": e.size_bytes / 1e9,
                    "cached_at": e.cached_at,
                    "last_accessed": e.last_accessed or e.cached_at,
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

