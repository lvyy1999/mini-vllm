from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

import minivllm.engine.model_runner as model_runner_module
from minivllm.engine.model_runner import ModelRunner
from minivllm.engine.sequence import Sequence
from minivllm.sampling_parameters import SamplingParams
from minivllm.utils.context import get_context, reset_context, set_context


@pytest.fixture(autouse=True)
def clean_context():
    original_block_size = Sequence.block_size
    Sequence.block_size = 4
    reset_context()
    yield
    reset_context()
    Sequence.block_size = original_block_size


@pytest.fixture
def cpu_tensor_transfers(monkeypatch):
    real_tensor = torch.tensor

    def make_cpu_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(model_runner_module.torch, "tensor", make_cpu_tensor)
    monkeypatch.setattr(
        torch.Tensor,
        "cuda",
        lambda tensor, non_blocking=False: tensor,
        raising=True,
    )


def make_runner_without_init(**attributes) -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    for name, value in attributes.items():
        setattr(runner, name, value)
    return runner


class TestSharedMemoryRPC:

    def test_write_and_read_round_trip(self):
        shared_memory = SimpleNamespace(buf=bytearray(1024))
        writer_events = [MagicMock(), MagicMock()]
        writer = make_runner_without_init(
            world_size=3,
            rank=0,
            shm=shared_memory,
            event=writer_events,
        )

        writer.write_shm("run", [1, 2], True)

        for event in writer_events:
            event.set.assert_called_once_with()

        reader_event = MagicMock()
        reader = make_runner_without_init(
            world_size=3,
            rank=1,
            shm=shared_memory,
            event=reader_event,
        )
        method_name, args = reader.read_shm()

        assert method_name == "run"
        assert args == [[1, 2], True]
        reader_event.wait.assert_called_once_with()
        reader_event.clear.assert_called_once_with()

    def test_write_requires_multi_gpu_rank_zero(self):
        runner = make_runner_without_init(world_size=2, rank=1)

        with pytest.raises(AssertionError, match="rank == 0"):
            runner.write_shm("run")

    def test_read_requires_nonzero_worker_rank(self):
        runner = make_runner_without_init(world_size=2, rank=0)

        with pytest.raises(AssertionError, match="rank != 0"):
            runner.read_shm()

    def test_call_broadcasts_from_rank_zero_then_invokes_local_method(self):
        runner = make_runner_without_init(world_size=2, rank=0)
        runner.write_shm = MagicMock()
        runner.add = MagicMock(return_value=7)

        result = runner.call("add", 3, 4)

        assert result == 7
        runner.write_shm.assert_called_once_with("add", 3, 4)
        runner.add.assert_called_once_with(3, 4)

    def test_unknown_method_is_rejected(self):
        runner = make_runner_without_init(world_size=1, rank=0)

        with pytest.raises(ValueError, match="Unknown method: missing"):
            runner.call("missing")

    def test_worker_loop_stops_after_exit_message(self):
        runner = make_runner_without_init(world_size=2, rank=1)
        runner.read_shm = MagicMock(
            side_effect=[("run", [[1], False]), ("exit", [])]
        )
        runner.call = MagicMock()

        runner.loop()

        assert runner.call.call_args_list == [
            call("run", [1], False),
            call("exit"),
        ]


class TestInputPreparation:

    def test_prepare_block_tables_pads_with_minus_one(self, cpu_tensor_transfers):
        first = Sequence([1])
        first.block_table = [2, 4]
        second = Sequence([2])
        second.block_table = [7]

        block_tables = ModelRunner.prepare_block_tables([first, second])

        assert block_tables.dtype == torch.int32
        assert block_tables.tolist() == [[2, 4], [7, -1]]

    def test_prepare_prefill_builds_chunked_attention_metadata(
        self, cpu_tensor_transfers
    ):
        runner = make_runner_without_init(block_size=4)
        first = Sequence([10, 11, 12])
        first.block_table = [2]
        first.num_scheduled_tokens = 3
        second = Sequence([20, 21, 22, 23, 24, 25])
        second.block_table = [5, 6]
        second.num_cached_tokens = 4
        second.num_scheduled_tokens = 2

        input_ids, positions = runner.prepare_prefill([first, second])
        context = get_context()

        assert input_ids.tolist() == [10, 11, 12, 24, 25]
        assert positions.tolist() == [0, 1, 2, 4, 5]
        assert context.is_prefill is True
        assert context.cu_seqlens_q.tolist() == [0, 3, 5]
        assert context.cu_seqlens_k.tolist() == [0, 3, 9]
        assert context.max_seqlen_q == 3
        assert context.max_seqlen_k == 6
        assert context.slot_mapping.tolist() == [8, 9, 10, 24, 25]
        assert context.context_lens is None
        assert context.block_tables.tolist() == [[2, -1], [5, 6]]

    def test_prepare_decode_builds_slots_lengths_and_tables(
        self, cpu_tensor_transfers
    ):
        runner = make_runner_without_init(block_size=4)
        first = Sequence([1, 2, 3, 4, 5])
        first.block_table = [2, 4]
        second = Sequence([6, 7, 8, 9, 10, 11, 12])
        second.block_table = [1, 3]

        input_ids, positions = runner.prepare_decode([first, second])
        context = get_context()

        assert input_ids.tolist() == [5, 12]
        assert positions.tolist() == [4, 6]
        assert context.is_prefill is False
        assert context.slot_mapping.tolist() == [16, 14]
        assert context.context_lens.tolist() == [5, 7]
        assert context.block_tables.tolist() == [[2, 4], [1, 3]]

    def test_prepare_sample_keeps_per_sequence_temperatures(
        self, cpu_tensor_transfers
    ):
        sequences = [
            Sequence([1], SamplingParams(temperature=0.5)),
            Sequence([2], SamplingParams(temperature=1.5)),
        ]

        temperatures = ModelRunner.prepare_sample(sequences)

        assert temperatures.dtype == torch.float32
        torch.testing.assert_close(temperatures, torch.tensor([0.5, 1.5]))


class TestModelExecution:

    def test_prefill_uses_eager_model_path(self):
        runner = make_runner_without_init(
            enforce_eager=False,
            max_graph_bs=8,
            model=MagicMock(),
        )
        hidden_states = torch.tensor([[1.0, 2.0]])
        logits = torch.tensor([[3.0, 4.0]])
        runner.model.return_value = hidden_states
        runner.model.compute_logits.return_value = logits
        input_ids = torch.tensor([1])
        positions = torch.tensor([0])

        output = runner.run_model(input_ids, positions, is_prefill=True)

        assert output is logits
        runner.model.assert_called_once_with(input_ids, positions)
        runner.model.compute_logits.assert_called_once_with(hidden_states)

    def test_decode_replays_smallest_fitting_graph(self):
        graph = MagicMock()
        graph_outputs = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        runner = make_runner_without_init(
            enforce_eager=False,
            max_graph_bs=4,
            graph_bs=[1, 2, 4],
            graphs={4: graph},
            graph_vars={
                "input_ids": torch.zeros(4, dtype=torch.long),
                "positions": torch.zeros(4, dtype=torch.long),
                "slot_mapping": torch.zeros(4, dtype=torch.int32),
                "context_lens": torch.zeros(4, dtype=torch.int32),
                "block_tables": torch.zeros(4, 3, dtype=torch.int32),
                "outputs": graph_outputs,
            },
            model=MagicMock(),
        )
        expected_logits = torch.tensor([[9.0]])
        runner.model.compute_logits.return_value = expected_logits
        input_ids = torch.tensor([10, 11, 12])
        positions = torch.tensor([4, 5, 6])
        set_context(
            is_prefill=False,
            slot_mapping=torch.tensor([20, 21, 22], dtype=torch.int32),
            context_lens=torch.tensor([5, 6, 7], dtype=torch.int32),
            block_tables=torch.tensor(
                [[1, 2], [3, -1], [4, 5]], dtype=torch.int32
            ),
        )

        output = runner.run_model(input_ids, positions, is_prefill=False)

        assert output is expected_logits
        graph.replay.assert_called_once_with()
        assert runner.graph_vars["input_ids"].tolist()[:3] == [10, 11, 12]
        assert runner.graph_vars["positions"].tolist()[:3] == [4, 5, 6]
        assert runner.graph_vars["slot_mapping"].tolist() == [20, 21, 22, -1]
        assert runner.graph_vars["context_lens"].tolist() == [5, 6, 7, 0]
        assert runner.graph_vars["block_tables"][:3, :2].tolist() == [
            [1, 2],
            [3, -1],
            [4, 5],
        ]
        runner.model.compute_logits.assert_called_once()
        torch.testing.assert_close(
            runner.model.compute_logits.call_args.args[0], graph_outputs[:3]
        )


class TestRunOrchestration:

    def test_rank_zero_prepares_runs_samples_and_resets_context(self):
        runner = make_runner_without_init(rank=0)
        sequences = [Sequence([1]), Sequence([2])]
        input_ids = torch.tensor([1, 2])
        positions = torch.tensor([0, 0])
        temperatures = torch.tensor([0.5, 1.0])
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        runner.prepare_prefill = MagicMock(return_value=(input_ids, positions))
        runner.prepare_decode = MagicMock()
        runner.prepare_sample = MagicMock(return_value=temperatures)
        runner.run_model = MagicMock(return_value=logits)
        runner.sampler = MagicMock(return_value=torch.tensor([1, 0]))
        set_context(is_prefill=True, max_seqlen_q=2)

        token_ids = runner.run(sequences, is_prefill=True)

        assert token_ids == [1, 0]
        runner.prepare_prefill.assert_called_once_with(sequences)
        runner.prepare_decode.assert_not_called()
        runner.prepare_sample.assert_called_once_with(sequences)
        runner.run_model.assert_called_once_with(input_ids, positions, True)
        runner.sampler.assert_called_once_with(logits, temperatures)
        assert get_context().is_prefill is False
        assert get_context().max_seqlen_q == 0

    def test_worker_rank_skips_sampling(self):
        runner = make_runner_without_init(rank=1)
        sequences = [Sequence([1])]
        input_ids = torch.tensor([1])
        positions = torch.tensor([0])
        runner.prepare_decode = MagicMock(return_value=(input_ids, positions))
        runner.prepare_sample = MagicMock()
        runner.run_model = MagicMock(return_value=torch.tensor([[1.0]]))
        runner.sampler = MagicMock()

        token_ids = runner.run(sequences, is_prefill=False)

        assert token_ids is None
        runner.prepare_sample.assert_not_called()
        runner.sampler.assert_not_called()
