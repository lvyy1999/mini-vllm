"""
Compare mini-vLLM with FP16, BF16, and INT8 KV caches.

Each mode runs in a fresh subprocess so CUDA memory, NCCL state, CUDA Graphs,
and the paged cache pool from one mode cannot affect another mode.

FP16 uses an FP16 model and FP16 KV cache. BF16 uses a BF16 model and BF16 KV
cache. INT8 keeps model computation in BF16 by default and quantizes only the
KV cache. Therefore, INT8 versus BF16 is the primary apples-to-apples
comparison. Use --int8-model-dtype float16 on GPUs without BF16 support.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
MODE_ORDER = ("fp16", "bf16", "int8")
MIB = 1024**2

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "max_position_embeddings": 40960,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "vocab_size": 151936,
}

DEFAULT_QUALITY_CASES = [
    {
        "prompt": "What is 17 multiplied by 23? Answer with only the number.",
        "accepted_answers": ["391"],
    },
    {
        "prompt": (
            "Which city is the capital of France? Answer with only the city name."
        ),
        "accepted_answers": ["Paris"],
    },
    {
        "prompt": "How many days are in a leap year? Answer with only the number.",
        "accepted_answers": ["366"],
    },
    {
        "prompt": (
            "What is the chemical formula for water? Answer with only the formula."
        ),
        "accepted_answers": ["H2O"],
    },
    {
        "prompt": ("What is 2 raised to the power of 10? Answer with only the number."),
        "accepted_answers": ["1024"],
    },
    {
        "prompt": (
            "Translate the English word 'hello' into Spanish. "
            "Answer with only one word."
        ),
        "accepted_answers": ["hola"],
    },
    {
        "prompt": "What is the square root of 144? Answer with only the number.",
        "accepted_answers": ["12"],
    },
    {
        "prompt": (
            "Which planet is the largest in the Solar System? "
            "Answer with only the planet name."
        ),
        "accepted_answers": ["Jupiter"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare mini-vLLM FP16, BF16, and INT8 KV-cache throughput, "
            "GPU memory, cache capacity, and generation quality."
        )
    )
    parser.add_argument(
        "--model-name-or-path",
        default=MODEL_NAME,
        help=(
            "Qwen3-0.6B Hugging Face name or local directory. "
            "The current benchmark model config is Qwen3-0.6B-specific."
        ),
    )
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--num-sequences", type=int, default=32)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cache-block-size", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument(
        "--random-length",
        action="store_true",
        help=(
            "Sample each throughput request's input length and max_tokens "
            "independently from one eighth of the maximum through the maximum."
        ),
    )
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help="Allow throughput requests to stop at EOS instead of forcing the limit.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable mini-vLLM CUDA Graph capture.",
    )
    parser.add_argument(
        "--int8-model-dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
        help=(
            "Model/activation dtype used with the INT8 KV cache. "
            "The matching FP16 or BF16 mode becomes the primary reference."
        ),
    )
    parser.add_argument(
        "--quality-max-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens for each quality prompt.",
    )
    parser.add_argument(
        "--quality-temperature",
        type=float,
        default=0.1,
        help="Low nonzero sampling temperature used for quality comparisons.",
    )
    parser.add_argument(
        "--quality-prompts-file",
        type=Path,
        help=(
            "Optional JSON list of prompt strings or objects with prompt and "
            "accepted_answers fields."
        ),
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=2,
        help="Number of quality prompts whose completions are printed.",
    )

    parser.add_argument(
        "--_worker-mode",
        choices=MODE_ORDER,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--_worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_result-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    internal_values = (
        args._worker_mode,
        args._worker_config,
        args._result_file,
    )
    if any(internal_values) and not all(internal_values):
        parser.error("internal worker arguments must be provided together")
    if args._worker_mode:
        return args

    if args.max_input_tokens <= 0:
        parser.error("--max-input-tokens must be greater than 0")
    if args.num_sequences <= 0:
        parser.error("--num-sequences must be greater than 0")
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be greater than 0")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be at least 0")
    if args.repeat <= 0:
        parser.error("--repeat must be greater than 0")
    if args.temperature <= 1e-10:
        parser.error("--temperature must be greater than 1e-10")
    if args.quality_temperature <= 1e-10:
        parser.error("--quality-temperature must be greater than 1e-10")
    if args.quality_max_tokens <= 0:
        parser.error("--quality-max-tokens must be greater than 0")
    if args.show_samples < 0:
        parser.error("--show-samples must be at least 0")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_num_batched_tokens <= 0:
        parser.error("--max-num-batched-tokens must be greater than 0")
    if args.cache_block_size <= 0 or args.cache_block_size & (
        args.cache_block_size - 1
    ):
        parser.error("--cache-block-size must be a positive power of two")
    if (
        args.max_input_tokens + args.max_output_tokens
        > MODEL_CONFIG["max_position_embeddings"]
    ):
        parser.error(
            "--max-input-tokens + --max-output-tokens must not exceed "
            f"{MODEL_CONFIG['max_position_embeddings']}"
        )
    return args


def load_quality_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(case) for case in DEFAULT_QUALITY_CASES]

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("quality prompt JSON must be a non-empty list")

    cases = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            prompt = item
            accepted_answers = []
        elif isinstance(item, dict):
            prompt = item.get("prompt")
            accepted_answers = item.get(
                "accepted_answers",
                item.get("answer", []),
            )
            if isinstance(accepted_answers, str):
                accepted_answers = [accepted_answers]
        else:
            raise ValueError(f"quality prompt item {index} must be a string or object")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"quality prompt item {index} has no valid prompt")
        if not isinstance(accepted_answers, list) or not all(
            isinstance(answer, str) and answer for answer in accepted_answers
        ):
            raise ValueError(
                f"quality prompt item {index} has invalid accepted_answers"
            )
        cases.append(
            {
                "prompt": prompt,
                "accepted_answers": accepted_answers,
            }
        )
    return cases


def build_parent_config(
    args: argparse.Namespace,
    quality_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_name_or_path": args.model_name_or_path,
        "max_input_tokens": args.max_input_tokens,
        "num_sequences": args.num_sequences,
        "max_output_tokens": args.max_output_tokens,
        "warmup_steps": args.warmup_steps,
        "repeat": args.repeat,
        "seed": args.seed,
        "temperature": args.temperature,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cache_block_size": args.cache_block_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "random_length": args.random_length,
        "respect_eos": args.respect_eos,
        "enforce_eager": args.enforce_eager,
        "int8_model_dtype": args.int8_model_dtype,
        "quality_max_tokens": args.quality_max_tokens,
        "quality_temperature": args.quality_temperature,
        "quality_cases": quality_cases,
    }


def set_seed(torch_module: Any, seed: int) -> None:
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)


def cuda_sync(torch_module: Any) -> None:
    torch_module.cuda.synchronize()


def mode_config(mode: str, int8_model_dtype: str) -> tuple[str, str]:
    if mode == "fp16":
        return "float16", "auto"
    if mode == "bf16":
        return "bfloat16", "auto"
    return int8_model_dtype, "int8"


def build_random_token_workload_batches(
    tokenizer: Any,
    max_input_tokens: int,
    max_output_tokens: int,
    num_sequences: int,
    num_batches: int,
    seed: int,
    random_length: bool,
) -> tuple[list[list[list[int]]], list[list[int]]]:
    special_token_ids = set(tokenizer.all_special_ids)
    candidate_token_ids = sorted(
        {
            token_id
            for token_id in tokenizer.get_vocab().values()
            if 0 <= token_id < MODEL_CONFIG["vocab_size"]
            and token_id not in special_token_ids
        }
    )
    if not candidate_token_ids:
        raise RuntimeError("tokenizer has no non-special token IDs to sample")

    rng = random.Random(seed)
    min_input_tokens = (max_input_tokens + 7) // 8
    min_output_tokens = (max_output_tokens + 7) // 8
    prompt_batches = []
    output_limit_batches = []
    for _ in range(num_batches):
        prompts = []
        output_limits = []
        for _ in range(num_sequences):
            if random_length:
                input_length = rng.randint(min_input_tokens, max_input_tokens)
                output_limit = rng.randint(
                    min_output_tokens,
                    max_output_tokens,
                )
            else:
                input_length = max_input_tokens
                output_limit = max_output_tokens
            prompts.append(rng.choices(candidate_token_ids, k=input_length))
            output_limits.append(output_limit)
        prompt_batches.append(prompts)
        output_limit_batches.append(output_limits)
    return prompt_batches, output_limit_batches


def tokenize_quality_cases(
    tokenizer: Any,
    quality_cases: list[dict[str, Any]],
) -> list[list[int]]:
    prompts = []
    for case in quality_cases:
        messages = [{"role": "user", "content": case["prompt"]}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        token_ids = tokenizer.encode(
            formatted_prompt,
            add_special_tokens=False,
        )
        if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) for token_id in token_ids
        ):
            raise TypeError(
                "quality prompt tokenization must return a list of integer IDs"
            )
        prompts.append(token_ids)
    return prompts


def normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def answer_is_accepted(text: str, accepted_answers: list[str]) -> bool | None:
    if not accepted_answers:
        return None
    normalized_text = normalize_answer(text)
    for answer in accepted_answers:
        normalized_answer = normalize_answer(answer)
        prefix = r"(?<!\w)" if normalized_answer[0].isalnum() else ""
        suffix = r"(?!\w)" if normalized_answer[-1].isalnum() else ""
        if re.search(
            prefix + re.escape(normalized_answer) + suffix,
            normalized_text,
        ):
            return True
    return False


def summarize_measurements(
    measurements: list[dict[str, float]],
) -> dict[str, float | int]:
    total_latency = sum(item["latency"] for item in measurements)
    total_tokens = int(sum(item["tokens"] for item in measurements))
    repeat = len(measurements)
    return {
        "total_latency_s": total_latency,
        "total_output_tokens": total_tokens,
        "avg_latency_s": total_latency / repeat,
        "avg_output_tokens": total_tokens / repeat,
        "tps": total_tokens / total_latency,
        "repeat": repeat,
    }


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "avg": sum(values) / len(values),
        "max": max(values),
    }


def workload_kv_upper_bound_bytes(
    prompt_batches: list[list[list[int]]],
    output_limit_batches: list[list[int]],
    block_size: int,
    block_bytes: int,
) -> int:
    per_batch_bytes = []
    for prompts, output_limits in zip(prompt_batches, output_limit_batches):
        blocks = sum(
            math.ceil((len(prompt) + output_limit) / block_size)
            for prompt, output_limit in zip(prompts, output_limits)
        )
        per_batch_bytes.append(blocks * block_bytes)
    return max(per_batch_bytes)


def run_worker(mode: str, config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from minivllm.llm import LLM
    from minivllm.sampling_parameters import SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_int8.py requires a CUDA-capable GPU")

    model_dtype, requested_kv_cache_dtype = mode_config(
        mode,
        config["int8_model_dtype"],
    )
    major, _ = torch.cuda.get_device_capability()
    bf16_supported = (
        torch.cuda.is_bf16_supported()
        if hasattr(torch.cuda, "is_bf16_supported")
        else major >= 8
    )
    if model_dtype == "bfloat16" and not bf16_supported:
        raise RuntimeError(
            f"{mode} mode requires native BF16 support; "
            "use a BF16-capable GPU or --int8-model-dtype float16 "
            "for the INT8 mode"
        )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    prompt_batches, output_limit_batches = build_random_token_workload_batches(
        tokenizer=tokenizer,
        max_input_tokens=config["max_input_tokens"],
        max_output_tokens=config["max_output_tokens"],
        num_sequences=config["num_sequences"],
        num_batches=config["warmup_steps"] + config["repeat"],
        seed=config["seed"],
        random_length=config["random_length"],
    )
    quality_prompt_ids = tokenize_quality_cases(
        tokenizer,
        config["quality_cases"],
    )

    quality_required_length = max(
        len(prompt) + config["quality_max_tokens"] for prompt in quality_prompt_ids
    )
    max_model_length = max(
        config["max_input_tokens"] + config["max_output_tokens"],
        quality_required_length,
    )
    if max_model_length > MODEL_CONFIG["max_position_embeddings"]:
        raise ValueError(
            "quality prompt length plus --quality-max-tokens exceeds "
            f"{MODEL_CONFIG['max_position_embeddings']}"
        )

    custom_model_config = dict(MODEL_CONFIG)
    custom_model_config["torch_dtype"] = model_dtype
    max_num_sequences = max(
        8,
        config["num_sequences"],
        len(quality_prompt_ids),
    )
    llm = LLM(
        model_name_or_path=config["model_name_or_path"],
        custom_model_config=custom_model_config,
        kv_cache_dtype=requested_kv_cache_dtype,
        gpu_memory_utilization=config["gpu_memory_utilization"],
        cache_block_size=config["cache_block_size"],
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_num_sequences=max_num_sequences,
        max_model_length=max_model_length,
        enforce_eager=config["enforce_eager"],
    )

    runner = llm.model_runner
    device = torch.cuda.current_device()
    cuda_sync(torch)
    engine_allocated_bytes = torch.cuda.memory_allocated(device)
    engine_reserved_bytes = torch.cuda.memory_reserved(device)

    data_bytes_per_vector = runner.head_dim * runner.kv_cache_dtype.itemsize
    scale_bytes_per_vector = torch.float32.itemsize if runner.kv_cache_int8 else 0
    kv_bytes_per_token = (
        2
        * runner.num_layers
        * runner.num_kv_heads
        * (data_bytes_per_vector + scale_bytes_per_vector)
    )
    kv_block_bytes = kv_bytes_per_token * runner.block_size
    max_cache_blocks = int(runner.config.max_cache_blocks)
    kv_pool_bytes = kv_block_bytes * max_cache_blocks
    cache_capacity_tokens = max_cache_blocks * runner.block_size

    sampling_batches = [
        [
            SamplingParams(
                temperature=config["temperature"],
                max_tokens=output_limit,
                ignore_eos=not config["respect_eos"],
            )
            for output_limit in output_limits
        ]
        for output_limits in output_limit_batches
    ]

    for index, (prompts, sampling_params) in enumerate(
        zip(
            prompt_batches[: config["warmup_steps"]],
            sampling_batches[: config["warmup_steps"]],
        )
    ):
        set_seed(torch, config["seed"] + 100_000 + index)
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cuda_sync(torch)

    cuda_sync(torch)
    steady_allocated_bytes = torch.cuda.memory_allocated(device)
    steady_reserved_bytes = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    measurements = []
    timed_prompt_batches = prompt_batches[config["warmup_steps"] :]
    timed_output_limit_batches = output_limit_batches[config["warmup_steps"] :]
    timed_sampling_batches = sampling_batches[config["warmup_steps"] :]
    for index, (prompts, sampling_params) in enumerate(
        zip(timed_prompt_batches, timed_sampling_batches)
    ):
        set_seed(torch, config["seed"] + index)
        cuda_sync(torch)
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        cuda_sync(torch)
        latency = time.perf_counter() - start
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        measurements.append(
            {
                "latency": latency,
                "tokens": output_tokens,
            }
        )

    timed_peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    timed_peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    throughput = summarize_measurements(measurements)

    input_lengths = [
        len(prompt) for prompts in timed_prompt_batches for prompt in prompts
    ]
    output_limits = [
        output_limit for limits in timed_output_limit_batches for output_limit in limits
    ]
    workload_upper_bound = workload_kv_upper_bound_bytes(
        timed_prompt_batches,
        timed_output_limit_batches,
        runner.block_size,
        kv_block_bytes,
    )

    set_seed(torch, config["seed"] + 1_000_003)
    quality_sampling = [
        SamplingParams(
            temperature=config["quality_temperature"],
            max_tokens=config["quality_max_tokens"],
            ignore_eos=False,
        )
        for _ in quality_prompt_ids
    ]
    quality_outputs = llm.generate(
        quality_prompt_ids,
        quality_sampling,
        use_tqdm=False,
    )
    cuda_sync(torch)

    quality = []
    for case, output in zip(config["quality_cases"], quality_outputs):
        quality.append(
            {
                "prompt": case["prompt"],
                "accepted_answers": case["accepted_answers"],
                "text": output["text"],
                "token_ids": output["token_ids"],
                "accepted_answer_hit": answer_is_accepted(
                    output["text"],
                    case["accepted_answers"],
                ),
            }
        )

    properties = torch.cuda.get_device_properties(device)
    return {
        "mode": mode,
        "model_dtype": model_dtype,
        "kv_cache_dtype": str(runner.kv_cache_dtype).removeprefix("torch."),
        "gpu": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_mib": properties.total_memory / MIB,
        },
        "throughput": throughput,
        "memory": {
            "engine_allocated_mib": engine_allocated_bytes / MIB,
            "engine_reserved_mib": engine_reserved_bytes / MIB,
            "steady_allocated_mib": steady_allocated_bytes / MIB,
            "steady_reserved_mib": steady_reserved_bytes / MIB,
            "timed_peak_allocated_mib": timed_peak_allocated_bytes / MIB,
            "timed_peak_reserved_mib": timed_peak_reserved_bytes / MIB,
            "timed_peak_extra_allocated_mib": max(
                0,
                timed_peak_allocated_bytes - steady_allocated_bytes,
            )
            / MIB,
            "kv_bytes_per_token": kv_bytes_per_token,
            "kv_block_bytes": kv_block_bytes,
            "kv_pool_mib": kv_pool_bytes / MIB,
            "max_cache_blocks": max_cache_blocks,
            "cache_capacity_tokens": cache_capacity_tokens,
            "workload_kv_upper_bound_mib": workload_upper_bound / MIB,
        },
        "workload": {
            "length_mode": ("random" if config["random_length"] else "fixed"),
            "input_tokens": summarize_lengths(input_lengths),
            "output_token_limits": summarize_lengths(output_limits),
            "num_sequences": config["num_sequences"],
        },
        "quality": quality,
    }


def levenshtein_distance(left: list[int], right: list[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def generation_agreement(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> dict[str, float]:
    if len(candidate) != len(reference):
        raise ValueError("quality result lengths do not match")

    exact_matches = 0
    positional_matches = 0
    positional_total = 0
    edit_similarities = []
    lcp_ratios = []
    candidate_tokens = 0
    reference_tokens = 0

    for candidate_item, reference_item in zip(candidate, reference):
        candidate_ids = candidate_item["token_ids"]
        reference_ids = reference_item["token_ids"]
        exact_matches += candidate_ids == reference_ids
        denominator = max(len(candidate_ids), len(reference_ids), 1)
        positional_matches += sum(
            left == right for left, right in zip(candidate_ids, reference_ids)
        )
        positional_total += denominator

        distance = levenshtein_distance(candidate_ids, reference_ids)
        edit_similarities.append(1.0 - distance / denominator)

        lcp = 0
        for left, right in zip(candidate_ids, reference_ids):
            if left != right:
                break
            lcp += 1
        lcp_ratios.append(lcp / denominator)
        candidate_tokens += len(candidate_ids)
        reference_tokens += len(reference_ids)

    count = len(reference)
    return {
        "exact_match_pct": 100.0 * exact_matches / count,
        "token_position_agreement_pct": (100.0 * positional_matches / positional_total),
        "mean_edit_similarity_pct": (100.0 * sum(edit_similarities) / count),
        "mean_lcp_ratio_pct": 100.0 * sum(lcp_ratios) / count,
        "length_ratio_pct": (100.0 * candidate_tokens / max(reference_tokens, 1)),
    }


def task_accuracy(quality: list[dict[str, Any]]) -> float | None:
    scores = [
        item["accepted_answer_hit"]
        for item in quality
        if item["accepted_answer_hit"] is not None
    ]
    if not scores:
        return None
    return 100.0 * sum(scores) / len(scores)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    separator = "-+-".join("-" * width for width in widths)
    lines = [
        " | ".join(value.ljust(width) for value, width in zip(headers, widths)),
        separator,
    ]
    lines.extend(
        " | ".join(value.ljust(width) for value, width in zip(row, widths))
        for row in rows
    )
    return "\n".join(lines)


def percent_delta(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def format_delta(value: float, reference: float) -> str:
    return f"{percent_delta(value, reference):+.1f}%"


def format_optional_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def compact_sample(text: str, max_chars: int = 800) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_report(
    config: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference_mode = "bf16" if config["int8_model_dtype"] == "bfloat16" else "fp16"
    reference_quality = results[reference_mode]["quality"]
    quality_metrics = {}
    for mode in MODE_ORDER:
        quality_metrics[mode] = {
            "task_accuracy_pct": task_accuracy(results[mode]["quality"]),
            **generation_agreement(
                results[mode]["quality"],
                reference_quality,
            ),
        }
    return {
        "config": config,
        "reference_mode": reference_mode,
        "results": results,
        "quality_metrics": quality_metrics,
    }


def print_report(report: dict[str, Any], show_samples: int) -> None:
    results = report["results"]
    reference_mode = report["reference_mode"]
    reference = results[reference_mode]

    print("\n=== Configuration ===")
    print(
        "INT8 uses "
        f"{results['int8']['model_dtype']} model computation and INT8 KV cache."
    )
    print(
        f"Primary reference: {reference_mode.upper()} "
        "(same model/activation dtype as INT8)."
    )
    print(
        f"GPU: {reference['gpu']['name']} "
        f"(compute capability {reference['gpu']['compute_capability']})"
    )
    workload = reference["workload"]
    input_stats = workload["input_tokens"]
    output_stats = workload["output_token_limits"]
    print(
        f"Throughput workload: {workload['num_sequences']} sequences/run, "
        f"{workload['length_mode']} lengths; input min/avg/max "
        f"{input_stats['min']}/{input_stats['avg']:.1f}/{input_stats['max']}, "
        f"max_tokens min/avg/max "
        f"{output_stats['min']}/{output_stats['avg']:.1f}/"
        f"{output_stats['max']}."
    )

    throughput_rows = []
    for mode in MODE_ORDER:
        result = results[mode]
        metrics = result["throughput"]
        throughput_rows.append(
            [
                mode.upper(),
                result["model_dtype"],
                result["kv_cache_dtype"],
                f"{metrics['tps']:.2f}",
                format_delta(
                    metrics["tps"],
                    reference["throughput"]["tps"],
                ),
                f"{metrics['avg_latency_s']:.3f}",
                f"{metrics['total_output_tokens']:,}",
            ]
        )
    print("\n=== Throughput ===")
    print(
        format_table(
            [
                "Mode",
                "Model dtype",
                "KV dtype",
                "TPS",
                f"vs {reference_mode.upper()}",
                "Avg latency (s)",
                "Output tokens",
            ],
            throughput_rows,
        )
    )

    memory_rows = []
    for mode in MODE_ORDER:
        memory = results[mode]["memory"]
        memory_rows.append(
            [
                mode.upper(),
                f"{memory['timed_peak_allocated_mib']:.1f}",
                f"{memory['timed_peak_extra_allocated_mib']:.1f}",
                f"{memory['kv_bytes_per_token']:,}",
                f"{memory['kv_pool_mib']:.1f}",
                f"{memory['max_cache_blocks']:,}",
                f"{memory['cache_capacity_tokens']:,}",
                f"{memory['workload_kv_upper_bound_mib']:.1f}",
            ]
        )
    print("\n=== GPU Memory And KV Capacity (per GPU) ===")
    print(
        format_table(
            [
                "Mode",
                "Peak alloc MiB",
                "Timed extra MiB",
                "KV bytes/token",
                "KV pool MiB",
                "Max blocks",
                "Cache tokens",
                "Workload KV MiB",
            ],
            memory_rows,
        )
    )

    quality_rows = []
    for mode in MODE_ORDER:
        metrics = report["quality_metrics"][mode]
        quality_rows.append(
            [
                mode.upper(),
                format_optional_pct(metrics["task_accuracy_pct"]),
                f"{metrics['exact_match_pct']:.1f}%",
                f"{metrics['token_position_agreement_pct']:.1f}%",
                f"{metrics['mean_edit_similarity_pct']:.1f}%",
                f"{metrics['mean_lcp_ratio_pct']:.1f}%",
                f"{metrics['length_ratio_pct']:.1f}%",
            ]
        )
    print("\n=== Accuracy And Generation Agreement ===")
    print(
        format_table(
            [
                "Mode",
                "Task accuracy",
                "Exact vs ref",
                "Token agree",
                "Edit similarity",
                "LCP ratio",
                "Length ratio",
            ],
            quality_rows,
        )
    )

    int8 = results["int8"]
    int8_quality = report["quality_metrics"]["int8"]
    reference_quality = report["quality_metrics"][reference_mode]
    int8_memory = int8["memory"]
    reference_memory = reference["memory"]
    peak_memory_delta = format_delta(
        int8_memory["timed_peak_allocated_mib"],
        reference_memory["timed_peak_allocated_mib"],
    )
    kv_bytes_delta = format_delta(
        int8_memory["kv_bytes_per_token"],
        reference_memory["kv_bytes_per_token"],
    )
    cache_capacity_delta = format_delta(
        int8_memory["cache_capacity_tokens"],
        reference_memory["cache_capacity_tokens"],
    )
    print("\n=== INT8 Impact Against Matching Reference ===")
    print(
        "Throughput: "
        f"{format_delta(int8['throughput']['tps'], reference['throughput']['tps'])}"
    )
    print(f"Timed peak allocated GPU memory: {peak_memory_delta}")
    print(f"KV bytes per token: {kv_bytes_delta}")
    print(f"Maximum cache token capacity: {cache_capacity_delta}")
    if int8_quality["task_accuracy_pct"] is not None:
        task_delta = (
            int8_quality["task_accuracy_pct"] - reference_quality["task_accuracy_pct"]
        )
        print(f"Quality smoke-test accuracy: {task_delta:+.1f} percentage points")
    print(
        "Same-seed token-position agreement: "
        f"{int8_quality['token_position_agreement_pct']:.1f}%"
    )
    print(
        "\nMemory note: mini-vLLM preallocates as many KV blocks as the configured "
        "GPU memory budget permits. Peak allocated memory can therefore remain "
        "similar while INT8 substantially lowers KV bytes/token and increases "
        "cache token capacity."
    )
    print(
        "Quality note: task accuracy is a small accepted-answer smoke test. "
        "Agreement metrics measure output stability against the matching "
        "FP16/BF16 run under the same seed; they are not a substitute for a "
        "full downstream evaluation dataset."
    )

    sample_count = min(
        show_samples,
        len(results[reference_mode]["quality"]),
    )
    if sample_count:
        print("\n=== Quality Samples ===")
        for index in range(sample_count):
            prompt = results[reference_mode]["quality"][index]["prompt"]
            print(f"\nPrompt {index + 1}: {prompt}")
            for mode in MODE_ORDER:
                text = results[mode]["quality"][index]["text"]
                print(f"[{mode.upper()}]\n{compact_sample(text)}")


def run_parent(args: argparse.Namespace) -> None:
    quality_cases = load_quality_cases(args.quality_prompts_file)
    config = build_parent_config(args, quality_cases)
    results = {}

    with tempfile.TemporaryDirectory(prefix="minivllm-int8-benchmark-") as tmp:
        temporary_dir = Path(tmp)
        config_path = temporary_dir / "config.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for index, mode in enumerate(MODE_ORDER, start=1):
            result_path = temporary_dir / f"{mode}.json"
            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--_worker-mode",
                mode,
                "--_worker-config",
                str(config_path),
                "--_result-file",
                str(result_path),
            ]
            print(
                f"\n[{index}/{len(MODE_ORDER)}] Running {mode.upper()} mode "
                "in a fresh process...",
                flush=True,
            )
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{mode.upper()} worker failed with exit code "
                    f"{completed.returncode}"
                )
            if not result_path.is_file():
                raise RuntimeError(
                    f"{mode.upper()} worker did not produce a result file"
                )
            results[mode] = json.loads(result_path.read_text(encoding="utf-8"))

    report = build_report(config, results)
    print_report(report, args.show_samples)


def run_worker_entry(args: argparse.Namespace) -> None:
    config = json.loads(args._worker_config.read_text(encoding="utf-8"))
    result = run_worker(args._worker_mode, config)
    args._result_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args._worker_mode:
        run_worker_entry(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
