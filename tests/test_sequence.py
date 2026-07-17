import pytest

from minivllm.engine.sequence import Sequence, SequenceStatus
from minivllm.sampling_parameters import SamplingParams


@pytest.fixture(autouse=True)
def small_block_size():
    original = Sequence.block_size
    Sequence.block_size = 4
    yield
    Sequence.block_size = original


class TestSequenceInitialization:

    def test_copies_prompt_tokens_and_initializes_metadata(self):
        prompt = [10, 20, 30]
        seq = Sequence(prompt)
        prompt.append(40)

        assert seq.token_ids == [10, 20, 30]
        assert seq.prompt_token_ids == [10, 20, 30]
        assert seq.completion_token_ids == []
        assert seq.last_token == 30
        assert len(seq) == 3
        assert seq.num_prompt_tokens == 3
        assert seq.num_cached_tokens == 0
        assert seq.num_scheduled_tokens == 0
        assert seq.status == SequenceStatus.WAITING
        assert seq.is_prefill

    def test_copies_sampling_parameters(self):
        params = SamplingParams(temperature=0.7, max_tokens=12, ignore_eos=True)
        seq = Sequence([1], params)

        assert seq.temperature == 0.7
        assert seq.max_tokens == 12
        assert seq.ignore_eos is True

    def test_sequence_ids_are_unique_and_increasing(self):
        first = Sequence([1])
        second = Sequence([2])

        assert second.seq_id == first.seq_id + 1


class TestSequenceBlocks:

    def test_splits_full_and_partial_blocks(self):
        seq = Sequence([1, 2, 3, 4, 5, 6])

        assert seq.num_blocks == 2
        assert seq.last_block_num_tokens == 2
        assert seq.block(0) == [1, 2, 3, 4]
        assert seq.block(1) == [5, 6]

    def test_exact_multiple_has_a_full_last_block(self):
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8])

        assert seq.num_blocks == 2
        assert seq.last_block_num_tokens == 4
        assert seq.block(1) == [5, 6, 7, 8]

    @pytest.mark.parametrize("index", [-1, 2])
    def test_rejects_out_of_range_block_index(self, index):
        seq = Sequence([1, 2, 3, 4, 5])

        with pytest.raises(AssertionError, match="Block index"):
            seq.block(index)


class TestSequenceMutation:

    def test_append_updates_completion_properties(self):
        seq = Sequence([1, 2])

        seq.append_token(3)
        seq.append_token(4)

        assert seq.token_ids == [1, 2, 3, 4]
        assert seq.last_token == 4
        assert seq.num_tokens == 4
        assert seq.num_completion_tokens == 2
        assert seq.completion_token_ids == [3, 4]

    def test_finished_property_tracks_status(self):
        seq = Sequence([1])

        assert not seq.is_finished
        seq.status = SequenceStatus.FINISHED
        assert seq.is_finished


class TestSequenceTransferState:

    def test_prefill_state_contains_all_tokens(self):
        seq = Sequence([1, 2, 3])
        seq.block_table = [7]
        seq.num_scheduled_tokens = 3

        state = seq.__getstate__()
        restored = Sequence.__new__(Sequence)
        restored.__setstate__(state)

        assert state[-1] == [1, 2, 3]
        assert restored.token_ids == [1, 2, 3]
        assert restored.last_token == 3
        assert restored.block_table == [7]
        assert restored.block_size == 4

    def test_decode_state_contains_only_the_last_token(self):
        seq = Sequence([1, 2, 3])
        seq.is_prefill = False
        seq.block_table = [4]

        state = seq.__getstate__()
        restored = Sequence.__new__(Sequence)
        restored.__setstate__(state)

        assert state[-1] == 3
        assert restored.token_ids == [3]
        assert restored.last_token == 3
        assert restored.num_tokens == 3
        assert restored.num_prompt_tokens == 3
