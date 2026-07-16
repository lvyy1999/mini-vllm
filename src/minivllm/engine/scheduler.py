from collections import deque

from minivllm.engine.block_manager import BlockManager
from minivllm.engine.sequence import Sequence, SequenceStatus
from minivllm.utils.config import Config


class Scheduler:

    def __init__(self, config: Config):
        self.eos = config.eos_token_id
        self.num_blocks = config.max_cache_blocks
        self.block_size = config.cache_block_size
        self.max_num_sequences = config.max_num_sequences
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # block manager
        self.block_manager = BlockManager(self.num_blocks, self.block_size)
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0

    def add_sequence(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], int, bool]:
        # init params
        scheduled_seqs = []
        num_batched_tokens = 0

        # try schedule for prefilling from waiting queue if not exceeding limits
        while (
            self.waiting
            and len(scheduled_seqs) < self.max_num_sequences
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            # calculate how many tokens need to schedule in this sequence,
            # which equals to total_tokens - num_cached_tokens
            if (
                not seq.block_table
            ):  # first schedule, need to check by can_allocate and get num_cached_tokens
                num_cached_tokens = self.block_manager.can_allocate(seq)
                if num_cached_tokens == -1:  # can't allocate this sequence
                    break
                num_tokens = seq.num_tokens - num_cached_tokens
            else:  # not first schedule, has allocated, get num_cached_tokens from seq directly
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # not enough to schedule this sequence and scheduled_seqs not empty, break;
            # only allow chunked prefill for the first seq
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:  # allocate those sequences first into schedule
                self.block_manager.allocate(seq)
            # maybe chunked prefill if num_tokens > remaining
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # if this sequence is all scheduled, move it from waiting queue into running queue
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # add into scheduled sequences list
            scheduled_seqs.append(seq)

        if scheduled_seqs:  # prefill seqs not empty
            return scheduled_seqs, num_batched_tokens, True

        # try schedule for decoding from running queue if not exceeding limits
        while (
            self.running
            and len(scheduled_seqs) < self.max_num_sequences
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.running.popleft()
            # use can_append to check whether we can append one more token
            while not self.block_manager.can_append(seq):  # can't append
                if self.running:  # running queue is not empty
                    self.preempt(
                        self.running.pop()
                    )  # preempt a sequence from queue tail
                else:  # running is empty, no other sequence to deallocate
                    self.preempt(seq)  # preempt itself
                    break
            else:  # can append
                seq.is_prefill = False  # decode mode
                seq.num_scheduled_tokens = 1  # schedule one token for decode
                num_batched_tokens += 1
                self.block_manager.append(seq)  # append one token
                scheduled_seqs.append(seq)  # add into scheduled sequences

        # assert scheduled_seqs, "Should schedule at least one seq in decode stage."

        # re-add into running queue in the same order
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, num_batched_tokens, False

    # preempt a seq, deallocate its source, re-add it into waiting queue's head
    def preempt(self, seq: Sequence) -> None:
        seq.is_prefill = True
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(
        self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool
    ) -> None:
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)  # update hash value
            seq.num_cached_tokens += (
                seq.num_scheduled_tokens
            )  # scheduled tokens in last turn become cached
            seq.num_scheduled_tokens = 0  # reset num_scheduled_tokens
            if (
                is_prefill and seq.num_cached_tokens < seq.num_tokens
            ):  # chunked prefill not finish
                continue
            seq.append_token(
                token_id
            )  # prefill finish or decode, append generated token into seq
            # check whether finish: get EOS token or reach max_tokens(number of completion tokens) limit
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
