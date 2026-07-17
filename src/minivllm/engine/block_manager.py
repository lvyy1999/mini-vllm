from collections import deque

import numpy as np
import xxhash

from minivllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = -1
        self.ref_count = 0
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1
        self.ref_count = 1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        # number of tokens per block
        self.block_size = block_size
        # list of all blocks
        self.blocks = [Block(i) for i in range(num_blocks)]
        # hash to block id for prefix caching
        self.hash_to_block_id = {}
        # free block ids
        self.free_block_ids = deque(range(num_blocks))
        # used block ids
        self.used_block_ids = set()

    # given token_ids, compute the hash value
    # use prefix_hash_value to compute the hash in a context-sensitive way
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(
                prefix_hash_value.to_bytes(8, "little"),
            )
        h.update(
            np.array(token_ids, dtype=np.int32).tobytes(),
        )
        return h.intdigest()

    # allocate a free block, move it from free blocks to used blocks
    def _allocate_block(self) -> int:
        assert len(self.free_block_ids) > 0, "No free blocks available"
        # get a free block from the head of deque
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0, "The block want to allocate is already allocated"
        # delay to delete hash map info because the block maybe reuse
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    # deallocate a block, move it from used blocks to free blocks
    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # whether we can allocate some blocks for this full sequence
    # if could, return the number of cached tokens; otherwise, return -1
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        # how many blocks find in cache, maybe in use or free
        # not equal to needn't allocate
        num_blocks_find_in_cache = 0
        # how many blocks need to allocate, miss in cache or hit cache but not in use,
        # only those blocks hit in cache and in use needn't allocate
        num_blocks_need_allocate = seq.num_blocks
        # the last block cannot use cached block, because maybe need to write new token
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            # find in cache by prefix hash
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # maybe exist hash collision, so need to check token_ids
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break  # prefix cache missed
            # cache hit a block
            num_blocks_find_in_cache += 1
            # cache hit and the block is in use, needn't allocate
            if block_id in self.used_block_ids:
                num_blocks_need_allocate -= 1
            # else:
            # there maybe some blocks deallocated into free blocks but not reset,
            # and the token_ids of them matched with the prefix of this sequence,
            # so could reuse them from free blocks, but need also to allocate
        # if free blocks enough to allocate, return the number of cached tokens,
        # where num_cached_tokens = num_blocks_find_in_cache * self.block_size;
        # otherwise, return -1
        return (
            num_blocks_find_in_cache * self.block_size
            if len(self.free_block_ids) >= num_blocks_need_allocate
            else -1
        )

    def allocate(self, seq: Sequence) -> None:
        assert not seq.block_table, "The sequence is already allocated"
        h = -1
        cache_missed = False
        num_cached_blocks = 0
        for i in range(seq.num_blocks):
            if not cache_missed and i != seq.num_blocks - 1:  # the last block cannot use cached block
                token_ids = seq.block(i)
                # find in cache by prefix hash
                h = self.compute_hash(token_ids, h)
                block_id = self.hash_to_block_id.get(h, -1)
                # maybe exist hash collision, so need to check token_ids
                if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                    cache_missed = True  # cache missed
                    seq.block_table.append(self._allocate_block())
                else:
                    num_cached_blocks += 1
                    if block_id in self.used_block_ids:
                        # cache hit and the block is in use, needn't allocate
                        self.blocks[block_id].ref_count += 1
                    else:
                        # cache hit and the block is free but not reset,
                        # reuse it by move it from free to used
                        self.blocks[block_id].ref_count = 1
                        self.used_block_ids.add(block_id)
                        self.free_block_ids.remove(block_id)
                    seq.block_table.append(block_id)
            else:  # cache missed, or be the last block, need to allocate a new block
                seq.block_table.append(self._allocate_block())
        # calculate the number of cached tokens
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence) -> None:
        # update block information; later allocate, earlier deallocate
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # update sequence information
        seq.block_table.clear()
        seq.num_cached_tokens = 0

    # whether we can append a token to this sequence
    def can_append(self, seq: Sequence) -> bool:
        need_new_block = seq.num_tokens % self.block_size == 1
        return True if not need_new_block else len(self.free_block_ids) > 0

    def append(self, seq: Sequence) -> None:
        need_new_block = seq.num_tokens % self.block_size == 1
        if need_new_block:
            seq.block_table.append(self._allocate_block())

    # only compute the block's hash value when it is full and has been scheduled
    def hash_blocks(self, seq: Sequence) -> None:
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return  # no new full blocks need to calculate hash
        # get prefix hash before the start block
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        # update the prefix hash value of new full blocks and prefix hash map,
        # the end block is not full, needn't update
        for i in range(start, end):
            block_id = seq.block_table[i]
            block = self.blocks[block_id]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
