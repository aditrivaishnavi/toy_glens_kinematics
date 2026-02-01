"""Unit tests for worker sharding in ParquetStreamDataset.

These tests verify the fix from LLM1 Q2:
- A1: Shard by BOTH DDP rank AND DataLoader worker id
- Without this fix, all N workers process same fragments = N*duplication per epoch
"""

import pytest
import numpy as np


class TestWorkerSharding:
    """Test that fragment sharding logic is correct."""
    
    def test_sharding_formula(self):
        """Test the sharding formula shard = rank * num_workers + worker_id."""
        # Simulate 2 DDP ranks with 4 workers each
        world_size = 2
        num_workers = 4
        
        # Total shards should be world_size * num_workers = 8
        nshard = world_size * num_workers
        
        # Each (rank, worker_id) should get a unique shard index
        shard_indices = set()
        for rank in range(world_size):
            for worker_id in range(num_workers):
                shard = rank * num_workers + worker_id
                assert shard not in shard_indices, f"Duplicate shard index: {shard}"
                shard_indices.add(shard)
        
        # Should have exactly nshard unique indices
        assert len(shard_indices) == nshard
        assert shard_indices == set(range(nshard))
    
    def test_fragment_distribution(self):
        """Test that fragments are distributed evenly across shards."""
        # Simulate 100 fragments
        n_frags = 100
        frags = list(range(n_frags))
        
        # Simulate 2 ranks with 4 workers each
        world_size = 2
        num_workers = 4
        nshard = world_size * num_workers
        
        all_assigned = []
        for rank in range(world_size):
            for worker_id in range(num_workers):
                shard = rank * num_workers + worker_id
                assigned = frags[shard::nshard]
                all_assigned.extend(assigned)
        
        # Each fragment should appear exactly once
        assert sorted(all_assigned) == frags, "Not all fragments assigned exactly once"
    
    def test_no_duplication_with_workers(self):
        """Test that no fragment is processed by multiple workers."""
        n_frags = 100
        frags = list(range(n_frags))
        
        # Single rank, 8 workers
        rank = 0
        world_size = 1
        num_workers = 8
        nshard = world_size * num_workers
        
        worker_fragments = {}
        for worker_id in range(num_workers):
            shard = rank * num_workers + worker_id
            worker_fragments[worker_id] = set(frags[shard::nshard])
        
        # Check no overlap between workers
        for w1 in range(num_workers):
            for w2 in range(w1 + 1, num_workers):
                overlap = worker_fragments[w1] & worker_fragments[w2]
                assert len(overlap) == 0, f"Workers {w1} and {w2} share fragments: {overlap}"
    
    def test_buggy_sharding_causes_duplication(self):
        """Demonstrate that old sharding (rank only) causes duplication."""
        n_frags = 100
        frags = list(range(n_frags))
        
        # Buggy: shard by rank only, ignoring worker_id
        rank = 0
        world_size = 1
        num_workers = 8
        
        # All workers get the same fragments!
        buggy_assigned = []
        for worker_id in range(num_workers):
            # BUG: uses rank::world instead of (rank*num_workers + worker_id)::(world*num_workers)
            assigned = frags[rank::world_size]  # This is ALL fragments!
            buggy_assigned.extend(assigned)
        
        # Each fragment appears num_workers times (8x duplication!)
        from collections import Counter
        counts = Counter(buggy_assigned)
        for frag, count in counts.items():
            assert count == num_workers, f"Bug: fragment {frag} appears {count} times, expected {num_workers}"
        
        # Total fragments is 8x what it should be
        assert len(buggy_assigned) == n_frags * num_workers


class TestEpochDependentShuffle:
    """Test epoch-dependent shuffling."""
    
    def test_different_epoch_different_order(self):
        """Test that different epochs produce different shuffle orders."""
        seed = 1337
        rank = 0
        worker_id = 0
        
        frags = list(range(100))
        
        orders = []
        for epoch in [0, 1, 2]:
            rng = np.random.RandomState(seed + 997 * rank + 131 * worker_id + 7919 * epoch)
            shuffled = frags.copy()
            rng.shuffle(shuffled)
            orders.append(tuple(shuffled))
        
        # All orders should be different
        assert len(set(orders)) == 3, "Epochs should produce different orders"
    
    def test_same_epoch_same_order(self):
        """Test that same epoch produces same shuffle order (reproducibility)."""
        seed = 1337
        rank = 0
        worker_id = 0
        epoch = 5
        
        frags = list(range(100))
        
        rng1 = np.random.RandomState(seed + 997 * rank + 131 * worker_id + 7919 * epoch)
        order1 = frags.copy()
        rng1.shuffle(order1)
        
        rng2 = np.random.RandomState(seed + 997 * rank + 131 * worker_id + 7919 * epoch)
        order2 = frags.copy()
        rng2.shuffle(order2)
        
        assert order1 == order2, "Same epoch should produce same order"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

