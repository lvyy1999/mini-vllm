from unittest.mock import MagicMock

import pytest

from minivllm.engine.scheduler import Scheduler
from minivllm.engine.sequence import Sequence, SequenceStatus
from minivllm.sampling_parameters import SamplingParams
from minivllm.utils.config import Config


@pytest.fixture(autouse=True)
def restore_sequence_block_size():
    original = Sequence.block_size
    yield
    Sequence.block_size = original


def make_scheduler(
    max_num_batched_tokens=100,
    max_num_sequences=10,
    max_cached_blocks=100,
    block_size=256,
    eos_token_id=-1,
):
    Sequence.block_size = block_size
    config = Config(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_sequences=max_num_sequences,
        max_cache_blocks=max_cached_blocks,
        cache_block_size=block_size,
        eos_token_id=eos_token_id,
    )
    return Scheduler(config)


def inject_running(scheduler: Scheduler, *seqs: Sequence):
    """Put sequences directly into the running queue, bypassing prefill."""
    for seq in seqs:
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)


def all_tracked(scheduler: Scheduler, scheduled: list[Sequence]) -> set:
    """Return the set of all sequences the scheduler currently knows about."""
    return set(scheduler.running) | set(scheduler.waiting) | set(scheduled)


class TestBug2TokenLimitBreak:
    """
    Setup: 3 sequences in running, all can_append=True.
           max_num_batched_tokens=2, so only 2 fit per step.
    Expected after schedule(): seq_c should still be in running.
    Buggy behaviour: seq_c is popleft-ed, the limit is hit, and seq_c is
                     never restored.
    """

    def _run(self, scheduler: Scheduler):
        seq_a = Sequence([1, 2, 3])
        seq_b = Sequence([4, 5, 6])
        seq_c = Sequence([7, 8, 9])
        inject_running(scheduler, seq_a, seq_b, seq_c)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = True
        scheduler.block_manager.append.return_value = None

        scheduled, _, is_prefill = scheduler.schedule()

        return seq_a, seq_b, seq_c, scheduled, is_prefill

    def test_seq_count_is_correct(self):
        scheduler = make_scheduler(max_num_batched_tokens=2)
        _, _, seq_c, scheduled, is_prefill = self._run(scheduler)

        assert not is_prefill
        assert len(scheduled) == 2
        assert seq_c in scheduler.running, (
            "Bug 2: seq_c was removed before the token-limit break and was "
            "not restored"
        )

    def test_seq_count_limit_variant(self):
        """The same regression can be triggered by max_num_sequences."""
        scheduler = make_scheduler(max_num_sequences=2, max_num_batched_tokens=100)
        _, _, seq_c, scheduled, is_prefill = self._run(scheduler)

        assert not is_prefill
        assert len(scheduled) == 2
        assert seq_c in scheduler.running, (
            "Bug 2: seq_c was removed before the sequence-limit break and was "
            "not restored"
        )

    def test_no_sequence_is_lost(self):
        """Every input sequence remains tracked after scheduling."""
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq_a, seq_b, seq_c, scheduled, _ = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        for seq in (seq_a, seq_b, seq_c):
            assert seq in tracked, f"seq {seq.seq_id} disappeared from the scheduler"


class TestBug1CanAppendFailure:
    """
    Setup: 2 sequences in running, can_append returns False for the first.
    Expected: seq_a is retried after another sequence is preempted, or is
              preempted itself; neither sequence may disappear.
    """

    def _run(self, scheduler: Scheduler):
        seq_a = Sequence([1, 2, 3])
        seq_b = Sequence([4, 5, 6])
        inject_running(scheduler, seq_a, seq_b)

        mock_bm = MagicMock()
        mock_bm.can_append.side_effect = [False, True, True, True]
        mock_bm.append.return_value = None
        mock_bm.deallocate.return_value = None
        scheduler.block_manager = mock_bm

        scheduled, _, is_prefill = scheduler.schedule()
        return seq_a, seq_b, scheduled, is_prefill

    def test_seq_a_not_lost(self):
        scheduler = make_scheduler()
        seq_a, _, scheduled, _ = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        assert seq_a in tracked, (
            "Bug 1: seq_a was removed before can_append failed and was not "
            "restored"
        )

    def test_total_conservation(self):
        """Neither sequence may disappear during preemption."""
        scheduler = make_scheduler()
        seq_a, seq_b, scheduled, _ = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        assert seq_a in tracked
        assert seq_b in tracked


class TestSchedulerHappyPath:

    def test_prefill_scheduled_first(self):
        scheduler = make_scheduler(max_num_batched_tokens=100, max_cached_blocks=50)
        seq = Sequence([1, 2, 3, 4])
        scheduler.add_sequence(seq)

        scheduled, num_tokens, is_prefill = scheduler.schedule()

        assert is_prefill
        assert num_tokens == 4
        assert seq in scheduled
        assert seq in scheduler.running

    def test_all_running_seqs_scheduled_when_budget_allows(self):
        scheduler = make_scheduler(max_num_batched_tokens=10)
        seq_a = Sequence([1])
        seq_b = Sequence([2])
        inject_running(scheduler, seq_a, seq_b)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = True
        scheduler.block_manager.append.return_value = None

        scheduled, num_tokens, is_prefill = scheduler.schedule()

        assert not is_prefill
        assert num_tokens == 2
        assert scheduled == [seq_a, seq_b]
        assert list(scheduler.running) == [seq_a, seq_b]

    def test_preempt_only_seq_when_cant_append_and_running_empty(self):
        """The only running sequence is preempted when it cannot append."""
        scheduler = make_scheduler()
        seq = Sequence([1, 2])
        inject_running(scheduler, seq)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = False
        scheduler.block_manager.deallocate.return_value = None

        scheduled, num_tokens, is_prefill = scheduler.schedule()

        assert not is_prefill
        assert num_tokens == 0
        assert scheduled == []
        assert seq in scheduler.waiting
        assert seq.status == SequenceStatus.WAITING


class TestSchedulerQueueState:

    def test_new_scheduler_is_finished(self):
        scheduler = make_scheduler()

        assert scheduler.is_finished()

    def test_add_sequence_appends_to_waiting_queue(self):
        scheduler = make_scheduler()
        seq = Sequence([1, 2, 3])

        scheduler.add_sequence(seq)

        assert not scheduler.is_finished()
        assert list(scheduler.waiting) == [seq]
        assert list(scheduler.running) == []

    def test_preempt_resets_sequence_and_moves_it_to_waiting_head(self):
        scheduler = make_scheduler()
        older_waiting = Sequence([1])
        seq = Sequence([2])
        scheduler.waiting.append(older_waiting)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        scheduler.block_manager = MagicMock()

        scheduler.preempt(seq)

        assert list(scheduler.waiting) == [seq, older_waiting]
        assert seq.status == SequenceStatus.WAITING
        assert seq.is_prefill
        scheduler.block_manager.deallocate.assert_called_once_with(seq)


class TestChunkedPrefill:

    def test_only_first_sequence_may_use_chunked_prefill(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=5,
            max_cached_blocks=10,
            block_size=4,
        )
        long_seq = Sequence([1, 2, 3, 4, 5, 6])
        next_seq = Sequence([7, 8])
        scheduler.add_sequence(long_seq)
        scheduler.add_sequence(next_seq)

        scheduled, num_tokens, is_prefill = scheduler.schedule()

        assert is_prefill
        assert scheduled == [long_seq]
        assert num_tokens == 5
        assert long_seq.num_scheduled_tokens == 5
        assert list(scheduler.waiting) == [long_seq, next_seq]
        assert list(scheduler.running) == []

    def test_partially_prefilled_sequence_resumes_from_cached_position(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_cached_blocks=10,
            block_size=4,
        )
        seq = Sequence([1, 2, 3, 4, 5, 6])
        scheduler.add_sequence(seq)

        first_batch, first_tokens, _ = scheduler.schedule()
        scheduler.postprocess(first_batch, [99], is_prefill=True)
        second_batch, second_tokens, is_prefill = scheduler.schedule()

        assert first_tokens == 4
        assert second_batch == [seq]
        assert second_tokens == 2
        assert is_prefill
        assert seq.num_cached_tokens == 4
        assert seq.num_scheduled_tokens == 2
        assert seq in scheduler.running


class TestSchedulerPostprocess:

    def test_chunked_prefill_caches_tokens_without_appending_sample(self):
        scheduler = make_scheduler()
        scheduler.block_manager = MagicMock()
        seq = Sequence([1, 2, 3])
        seq.num_scheduled_tokens = 2

        scheduler.postprocess([seq], [99], is_prefill=True)

        scheduler.block_manager.hash_blocks.assert_called_once_with(seq)
        assert seq.num_cached_tokens == 2
        assert seq.num_scheduled_tokens == 0
        assert seq.token_ids == [1, 2, 3]

    def test_completed_prefill_appends_sampled_token(self):
        scheduler = make_scheduler()
        scheduler.block_manager = MagicMock()
        seq = Sequence([1, 2, 3])
        inject_running(scheduler, seq)
        seq.num_scheduled_tokens = 3

        scheduler.postprocess([seq], [4], is_prefill=True)

        assert seq.num_cached_tokens == 3
        assert seq.completion_token_ids == [4]
        assert seq.status == SequenceStatus.RUNNING

    def test_eos_finishes_sequence_and_releases_blocks(self):
        scheduler = make_scheduler(eos_token_id=2)
        scheduler.block_manager = MagicMock()
        seq = Sequence([1])
        inject_running(scheduler, seq)
        seq.num_scheduled_tokens = 1

        scheduler.postprocess([seq], [2], is_prefill=False)

        assert seq.is_finished
        assert seq.completion_token_ids == [2]
        assert seq not in scheduler.running
        scheduler.block_manager.deallocate.assert_called_once_with(seq)

    def test_ignore_eos_keeps_sequence_running(self):
        scheduler = make_scheduler(eos_token_id=2)
        scheduler.block_manager = MagicMock()
        params = SamplingParams(max_tokens=2, ignore_eos=True)
        seq = Sequence([1], params)
        inject_running(scheduler, seq)
        seq.num_scheduled_tokens = 1

        scheduler.postprocess([seq], [2], is_prefill=False)

        assert not seq.is_finished
        assert seq in scheduler.running
        scheduler.block_manager.deallocate.assert_not_called()

    def test_max_tokens_finishes_sequence(self):
        scheduler = make_scheduler(eos_token_id=99)
        scheduler.block_manager = MagicMock()
        seq = Sequence([1], SamplingParams(max_tokens=1))
        inject_running(scheduler, seq)
        seq.num_scheduled_tokens = 1

        scheduler.postprocess([seq], [2], is_prefill=False)

        assert seq.is_finished
        assert seq.num_completion_tokens == 1
        scheduler.block_manager.deallocate.assert_called_once_with(seq)
