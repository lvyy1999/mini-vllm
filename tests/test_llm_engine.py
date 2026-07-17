from unittest.mock import MagicMock, call

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("transformers")
pytest.importorskip("tqdm")

from minivllm.engine.llm_engine import LLMEngine
from minivllm.engine.sequence import Sequence, SequenceStatus
from minivllm.sampling_parameters import SamplingParams


def make_engine_without_init() -> LLMEngine:
    return LLMEngine.__new__(LLMEngine)


class TestEnginePromptHandling:

    def test_string_prompt_is_tokenized_before_enqueue(self):
        engine = make_engine_without_init()
        engine.tokenizer = MagicMock()
        engine.tokenizer.encode.return_value = [10, 20, 30]
        engine.scheduler = MagicMock()
        params = SamplingParams(temperature=0.7, max_tokens=12)

        engine.add_prompt("hello", params)

        engine.tokenizer.encode.assert_called_once_with("hello")
        sequence = engine.scheduler.add_sequence.call_args.args[0]
        assert sequence.token_ids == [10, 20, 30]
        assert sequence.temperature == 0.7
        assert sequence.max_tokens == 12

    def test_token_prompt_bypasses_tokenizer_and_is_copied(self):
        engine = make_engine_without_init()
        engine.tokenizer = MagicMock()
        engine.scheduler = MagicMock()
        prompt = [1, 2, 3]

        engine.add_prompt(prompt, SamplingParams())
        prompt.append(4)

        engine.tokenizer.encode.assert_not_called()
        sequence = engine.scheduler.add_sequence.call_args.args[0]
        assert sequence.token_ids == [1, 2, 3]

    def test_is_finished_delegates_to_scheduler(self):
        engine = make_engine_without_init()
        engine.scheduler = MagicMock()
        engine.scheduler.is_finished.return_value = True

        assert engine.is_finished() is True
        engine.scheduler.is_finished.assert_called_once_with()


class TestEngineStep:

    def test_runs_scheduler_model_and_postprocess_in_order(self):
        engine = make_engine_without_init()
        engine.scheduler = MagicMock()
        engine.model_runner = MagicMock()
        finished = Sequence([1])
        finished.append_token(10)
        finished.status = SequenceStatus.FINISHED
        running = Sequence([2])
        running.status = SequenceStatus.RUNNING
        sequences = [finished, running]
        engine.scheduler.schedule.return_value = (sequences, 2, False)
        engine.model_runner.call.return_value = [10, 11]

        outputs, num_tokens, is_prefill = engine.step()

        assert outputs == [(finished.seq_id, [10])]
        assert num_tokens == 2
        assert is_prefill is False
        engine.model_runner.call.assert_called_once_with("run", sequences, False)
        engine.scheduler.postprocess.assert_called_once_with(
            sequences, [10, 11], False
        )


class TestEngineGenerate:

    def test_collects_finished_outputs_in_sequence_id_order(self):
        engine = make_engine_without_init()
        engine.add_prompt = MagicMock()
        engine.is_finished = MagicMock(side_effect=[False, True])
        engine.step = MagicMock(
            return_value=([(1, [20, 21]), (0, [10])], 2, False)
        )
        engine.tokenizer = MagicMock()
        engine.tokenizer.decode.side_effect = lambda token_ids: ",".join(
            map(str, token_ids)
        )
        params = SamplingParams(max_tokens=2)

        outputs = engine.generate(["first", "second"], params, use_tqdm=False)

        assert outputs == [
            {"text": "10", "token_ids": [10]},
            {"text": "20,21", "token_ids": [20, 21]},
        ]
        assert engine.add_prompt.call_args_list == [
            call("first", params),
            call("second", params),
        ]

    def test_accepts_per_prompt_sampling_parameters(self):
        engine = make_engine_without_init()
        engine.add_prompt = MagicMock()
        engine.is_finished = MagicMock(return_value=True)
        engine.tokenizer = MagicMock()
        first_params = SamplingParams(temperature=0.5)
        second_params = SamplingParams(temperature=1.5)

        outputs = engine.generate(
            [[1], [2]], [first_params, second_params], use_tqdm=False
        )

        assert outputs == []
        assert engine.add_prompt.call_args_list == [
            call([1], first_params),
            call([2], second_params),
        ]


class TestEngineCleanup:

    def test_exit_stops_runner_and_joins_workers(self):
        engine = make_engine_without_init()
        model_runner = MagicMock()
        workers = [MagicMock(), MagicMock()]
        engine.model_runner = model_runner
        engine.processes = workers

        engine.exit()

        model_runner.call.assert_called_once_with("exit")
        for worker in workers:
            worker.join.assert_called_once_with()
        assert not hasattr(engine, "model_runner")
