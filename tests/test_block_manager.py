import pytest

from minivllm.engine.block_manager import BlockManager
from minivllm.engine.sequence import Sequence


@pytest.fixture(autouse=True)
def small_block_size():
    original = Sequence.block_size
    Sequence.block_size = 4
    yield
    Sequence.block_size = original


def mark_scheduled_tokens_cached(
    manager: BlockManager, seq: Sequence, num_tokens: int
) -> None:
    seq.num_scheduled_tokens = num_tokens
    manager.hash_blocks(seq)
    seq.num_cached_tokens += num_tokens
    seq.num_scheduled_tokens = 0


class TestBlockLifecycle:

    def test_initial_state_contains_only_free_blocks(self):
        manager = BlockManager(num_blocks=3, block_size=4)

        assert list(manager.free_block_ids) == [0, 1, 2]
        assert manager.used_block_ids == set()
        assert all(block.ref_count == 0 for block in manager.blocks)

    def test_allocate_and_deallocate_update_all_metadata(self):
        manager = BlockManager(num_blocks=3, block_size=4)
        seq = Sequence([1, 2, 3, 4, 5])

        assert manager.can_allocate(seq) == 0
        manager.allocate(seq)

        allocated = list(seq.block_table)
        assert allocated == [0, 1]
        assert manager.used_block_ids == {0, 1}
        assert list(manager.free_block_ids) == [2]
        assert all(manager.blocks[i].ref_count == 1 for i in allocated)

        manager.deallocate(seq)

        assert seq.block_table == []
        assert seq.num_cached_tokens == 0
        assert manager.used_block_ids == set()
        assert set(manager.free_block_ids) == {0, 1, 2}
        assert all(manager.blocks[i].ref_count == 0 for i in allocated)

    def test_reports_insufficient_capacity(self):
        manager = BlockManager(num_blocks=1, block_size=4)
        seq = Sequence([1, 2, 3, 4, 5])

        assert manager.can_allocate(seq) == -1

    def test_allocating_from_an_empty_pool_fails(self):
        manager = BlockManager(num_blocks=1, block_size=4)
        manager._allocate_block()

        with pytest.raises(AssertionError, match="No free blocks"):
            manager._allocate_block()


class TestAppendCapacity:

    def test_allocates_a_new_block_at_a_block_boundary(self):
        manager = BlockManager(num_blocks=2, block_size=4)
        seq = Sequence([1, 2, 3, 4])
        manager.allocate(seq)
        seq.append_token(5)

        assert manager.can_append(seq)
        manager.append(seq)

        assert len(seq.block_table) == 2
        assert len(manager.free_block_ids) == 0

    def test_cannot_cross_a_boundary_without_a_free_block(self):
        manager = BlockManager(num_blocks=1, block_size=4)
        seq = Sequence([1, 2, 3, 4])
        manager.allocate(seq)
        seq.append_token(5)

        assert not manager.can_append(seq)

    def test_does_not_allocate_inside_an_existing_block(self):
        manager = BlockManager(num_blocks=2, block_size=4)
        seq = Sequence([1, 2, 3, 4, 5])
        manager.allocate(seq)
        seq.append_token(6)
        original_table = list(seq.block_table)

        assert manager.can_append(seq)
        manager.append(seq)

        assert seq.block_table == original_table


class TestBlockHashing:

    def test_hashes_a_block_only_after_it_becomes_full(self):
        manager = BlockManager(num_blocks=2, block_size=4)
        seq = Sequence([1, 2, 3])
        manager.allocate(seq)

        mark_scheduled_tokens_cached(manager, seq, 3)
        block_id = seq.block_table[0]
        assert manager.blocks[block_id].hash == -1
        assert manager.hash_to_block_id == {}

        seq.append_token(4)
        mark_scheduled_tokens_cached(manager, seq, 1)

        block = manager.blocks[block_id]
        assert block.hash != -1
        assert block.token_ids == [1, 2, 3, 4]
        assert manager.hash_to_block_id[block.hash] == block_id

    def test_hash_includes_the_prefix_hash(self):
        tokens = [1, 2, 3, 4]

        without_prefix = BlockManager.compute_hash(tokens, -1)
        with_prefix = BlockManager.compute_hash(tokens, 123)

        assert without_prefix != with_prefix

    def test_hash_collision_still_checks_token_ids(self, monkeypatch):
        def constant_hash(cls, token_ids, prefix_hash_value):
            return 7

        monkeypatch.setattr(
            BlockManager, "compute_hash", classmethod(constant_hash)
        )
        manager = BlockManager(num_blocks=6, block_size=4)
        first = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
        manager.allocate(first)
        mark_scheduled_tokens_cached(manager, first, 8)

        second = Sequence([9, 10, 11, 12, 13, 14, 15, 16])

        assert manager.can_allocate(second) == 0
        manager.allocate(second)
        assert second.block_table[0] not in first.block_table


class TestPrefixCacheReuse:

    def test_reuses_a_complete_prefix_block_while_it_is_active(self):
        manager = BlockManager(num_blocks=5, block_size=4)
        first = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
        manager.allocate(first)
        mark_scheduled_tokens_cached(manager, first, 8)

        second = Sequence([1, 2, 3, 4, 9, 10, 11, 12])
        assert manager.can_allocate(second) == 4
        manager.allocate(second)

        shared_block = first.block_table[0]
        assert second.block_table[0] == shared_block
        assert second.block_table[-1] != first.block_table[-1]
        assert second.num_cached_tokens == 4
        assert manager.blocks[shared_block].ref_count == 2

    def test_reuses_a_cached_block_after_it_is_deallocated(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        first = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
        manager.allocate(first)
        mark_scheduled_tokens_cached(manager, first, 8)
        cached_block = first.block_table[0]
        manager.deallocate(first)

        assert cached_block in manager.free_block_ids

        second = Sequence([1, 2, 3, 4, 9, 10, 11, 12])
        manager.allocate(second)

        assert second.block_table[0] == cached_block
        assert cached_block in manager.used_block_ids
        assert cached_block not in manager.free_block_ids
        assert manager.blocks[cached_block].ref_count == 1

    def test_shared_block_is_freed_only_after_all_references_finish(self):
        manager = BlockManager(num_blocks=5, block_size=4)
        first = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
        manager.allocate(first)
        mark_scheduled_tokens_cached(manager, first, 8)
        second = Sequence([1, 2, 3, 4, 9, 10, 11, 12])
        manager.allocate(second)
        shared_block = first.block_table[0]

        manager.deallocate(first)
        assert manager.blocks[shared_block].ref_count == 1
        assert shared_block in manager.used_block_ids

        manager.deallocate(second)
        assert manager.blocks[shared_block].ref_count == 0
        assert shared_block not in manager.used_block_ids
        assert shared_block in manager.free_block_ids

    def test_final_block_is_never_shared_between_active_sequences(self):
        manager = BlockManager(num_blocks=3, block_size=4)
        first = Sequence([1, 2, 3, 4])
        manager.allocate(first)
        mark_scheduled_tokens_cached(manager, first, 4)

        second = Sequence([1, 2, 3, 4])
        assert manager.can_allocate(second) == 0
        manager.allocate(second)

        assert first.block_table[0] != second.block_table[0]
        assert manager.blocks[first.block_table[0]].ref_count == 1
        assert manager.blocks[second.block_table[0]].ref_count == 1
