from copy import copy
from enum import Enum, auto
from itertools import count

from minivllm.sampling_parameters import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()
    block_size = 256

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        # record sequence id
        self.seq_id = next(Sequence.counter)
        # sequence status
        self.status = SequenceStatus.WAITING
        # token ids, need copy so that it is a new list, won't be affected by outside changes
        self.token_ids = copy(token_ids)
        # init needed params
        self.last_token = self.token_ids[-1] if self.token_ids else None
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True  # prefill or decode
        self.block_table = []
        # sampling_params
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, idx):
        return self.token_ids[idx]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert (
            0 <= i < self.num_blocks
        ), f"Block index {i} out of range [0, {self.num_blocks})"
        if i == self.num_blocks - 1:  # the last block maybe not full
            return self.token_ids[-self.last_block_num_tokens :]
        else:
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            return self.token_ids[start_idx:end_idx]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            self.block_size,
            self.token_ids if self.is_prefill else self.last_token,
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            self.block_size,
            last_token_or_ids,
        ) = state
        if isinstance(last_token_or_ids, list):
            # Prefill: last_token_or_ids is the full token_ids list
            self.token_ids = last_token_or_ids
        else:
            # Decode: last_token_or_ids is just the last token
            self.token_ids = [last_token_or_ids]
        # Restore last_token attribute
        self.last_token = self.token_ids[-1] if self.token_ids else None
