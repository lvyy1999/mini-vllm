import argparse
import gc
import random
import sys
import time
from pathlib import Path

import torch
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ===== minivllm =====
from minivllm.llm import LLM as MiniVLLM
from minivllm.sampling_parameters import SamplingParams as MiniSamplingParams

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
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "vocab_size": 151936,
}

MAX_INPUT_TOKENS = 128
NUM_SEQUENCES = 3
WARMUP_STEPS = 2
MAX_OUTPUT_TOKENS = 256  # maximum output token count
REPEAT_STEPS = 1
SEED = 0
IGNORE_EOS = True
TEMPERATURE = 0.6
ENFORCE_EAGER = False
MODEL_DTYPE = MODEL_CONFIG["torch_dtype"]
device = "cuda" if torch.cuda.is_available() else "cpu"


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype_from_name(dtype: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def transformers_dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    dtype_arg = (
        "dtype"
        if Version(transformers.__version__) >= Version("4.56.0")
        else "torch_dtype"
    )
    return {dtype_arg: dtype}


def summarize_measurements(measurements):
    total_latency = sum(item["latency"] for item in measurements)
    total_tokens = sum(item["tokens"] for item in measurements)
    repeat = len(measurements)
    return {
        "total_latency": total_latency,
        "total_tokens": total_tokens,
        "avg_latency": total_latency / repeat,
        "avg_tokens": total_tokens / repeat,
        "tps": total_tokens / total_latency,
        "repeat": repeat,
    }


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_random_token_workload_batches(
    tokenizer,
    max_input_tokens: int,
    max_output_tokens: int,
    num_sequences: int,
    num_batches: int,
    seed: int,
    random_length: bool,
) -> tuple[list[list[list[int]]], list[list[int]]]:
    special_token_ids = set(tokenizer.all_special_ids)
    model_vocab_size = MODEL_CONFIG["vocab_size"]
    candidate_token_ids = sorted(
        {
            token_id
            for token_id in tokenizer.get_vocab().values()
            if 0 <= token_id < model_vocab_size
            and token_id not in special_token_ids
        }
    )
    if not candidate_token_ids:
        raise RuntimeError("Tokenizer has no non-special token IDs to sample from")

    rng = random.Random(seed)
    min_input_tokens = (max_input_tokens + 7) // 8
    min_output_tokens = (max_output_tokens + 7) // 8
    prompt_batches = []
    output_token_limit_batches = []
    for _ in range(num_batches):
        prompts = []
        output_token_limits = []
        for _ in range(num_sequences):
            if random_length:
                input_length = rng.randint(min_input_tokens, max_input_tokens)
                output_token_limit = rng.randint(
                    min_output_tokens, max_output_tokens
                )
            else:
                input_length = max_input_tokens
                output_token_limit = max_output_tokens
            prompts.append(rng.choices(candidate_token_ids, k=input_length))
            output_token_limits.append(output_token_limit)
        prompt_batches.append(prompts)
        output_token_limit_batches.append(output_token_limits)
    return prompt_batches, output_token_limit_batches


def build_left_padded_batch(prompts, pad_token_id):
    max_input_length = max(len(prompt) for prompt in prompts)
    input_ids = torch.full(
        (len(prompts), max_input_length),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompts):
        prompt_length = len(prompt)
        input_ids[row, -prompt_length:] = torch.tensor(prompt, dtype=torch.long)
        attention_mask[row, -prompt_length:] = 1
    return input_ids.to(device), attention_mask.to(device)


def run_minivllm(
    prompt_batches,
    output_token_limit_batches,
    random_length,
    gpu_memory_utilization=0.9,
):
    model_config = dict(MODEL_CONFIG)
    model_config["torch_dtype"] = MODEL_DTYPE
    llm = MiniVLLM(
        enforce_eager=ENFORCE_EAGER,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_length=MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS,
        model_name_or_path=MODEL_NAME,
        custom_model_config=model_config,
        cache_block_size=16,
    )

    if random_length:
        sampling_batches = [
            [
                MiniSamplingParams(
                    temperature=TEMPERATURE,
                    max_tokens=output_token_limit,
                    ignore_eos=IGNORE_EOS,
                )
                for output_token_limit in output_token_limits
            ]
            for output_token_limits in output_token_limit_batches
        ]
    else:
        sampling = MiniSamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            ignore_eos=IGNORE_EOS,
        )
        sampling_batches = [sampling] * len(prompt_batches)

    # warmup
    for prompts, sampling_params in zip(
        prompt_batches[:WARMUP_STEPS], sampling_batches[:WARMUP_STEPS]
    ):
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cuda_sync()

    measurements = []
    outputs = None
    timed_batches = zip(
        prompt_batches[WARMUP_STEPS:], sampling_batches[WARMUP_STEPS:]
    )
    for i, (prompts, sampling_params) in enumerate(timed_batches):
        set_seed(SEED + i)
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        cuda_sync()
        end = time.perf_counter()

        total_tokens = sum(len(output["token_ids"]) for output in outputs)
        latency = end - start
        measurements.append(
            {
                "latency": latency,
                "tokens": total_tokens,
                "tps": total_tokens / latency,
            }
        )

    result = summarize_measurements(measurements)
    del llm, outputs
    cleanup_cuda()
    return result


def run_vllm(
    prompt_batches,
    output_token_limit_batches,
    random_length,
    gpu_memory_utilization,
):
    try:
        from vllm import LLM as VLLM
        from vllm import SamplingParams as VLLMSamplingParams
    except ImportError as exc:
        return {"error": f"vLLM is not installed: {exc}"}

    max_model_len = MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS
    vllm_prompt_batches = [
        [{"prompt_token_ids": prompt} for prompt in prompts]
        for prompts in prompt_batches
    ]

    # vLLM
    llm = VLLM(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        trust_remote_code=False,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        speculative_config=None,
        dtype=MODEL_DTYPE,
        enforce_eager=ENFORCE_EAGER,
        block_size=16,
    )

    if random_length:
        sampling_batches = [
            [
                VLLMSamplingParams(
                    temperature=TEMPERATURE,
                    max_tokens=output_token_limit,
                    ignore_eos=IGNORE_EOS,
                )
                for output_token_limit in output_token_limits
            ]
            for output_token_limits in output_token_limit_batches
        ]
    else:
        sampling = VLLMSamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            ignore_eos=IGNORE_EOS,
        )
        sampling_batches = [sampling] * len(vllm_prompt_batches)

    # warmup
    for prompts, sampling_params in zip(
        vllm_prompt_batches[:WARMUP_STEPS], sampling_batches[:WARMUP_STEPS]
    ):
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cuda_sync()

    measurements = []
    outputs = None
    timed_batches = zip(
        vllm_prompt_batches[WARMUP_STEPS:], sampling_batches[WARMUP_STEPS:]
    )
    for i, (prompts, sampling_params) in enumerate(timed_batches):
        set_seed(SEED + i)
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        cuda_sync()
        end = time.perf_counter()

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        latency = end - start
        measurements.append(
            {
                "latency": latency,
                "tokens": total_tokens,
                "tps": total_tokens / latency,
            }
        )

    result = summarize_measurements(measurements)
    del llm, outputs
    cleanup_cuda()
    return result


def run_transformers_test(tokenizer, prompt_batches):
    # transformers
    dtype_kwargs = transformers_dtype_kwargs(torch_dtype_from_name(MODEL_DTYPE))
    print(
        f"Transformers {transformers.__version__}: loading {MODEL_DTYPE} "
        f"with {next(iter(dtype_kwargs))}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, **dtype_kwargs
    ).to(device)
    model.eval()
    if IGNORE_EOS:
        model.generation_config.eos_token_id = None

    # warmup
    generation_kwargs = dict(
        max_new_tokens=MAX_OUTPUT_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        pad_token_id=tokenizer.pad_token_id,
    )
    if IGNORE_EOS:
        generation_kwargs["eos_token_id"] = None

    for prompts in prompt_batches[:WARMUP_STEPS]:
        input_ids, attention_mask = build_left_padded_batch(
            prompts, tokenizer.pad_token_id
        )
        with torch.inference_mode():
            model.generate(
                input_ids, attention_mask=attention_mask, **generation_kwargs
            )
        cuda_sync()

    measurements = []
    outputs = None
    for i, prompts in enumerate(prompt_batches[WARMUP_STEPS:]):
        input_ids, attention_mask = build_left_padded_batch(
            prompts, tokenizer.pad_token_id
        )
        cuda_sync()
        set_seed(SEED + i)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids, attention_mask=attention_mask, **generation_kwargs
            )
        cuda_sync()
        end = time.perf_counter()

        input_length = input_ids.shape[1]
        total_tokens = sum(max(0, len(output) - input_length) for output in outputs)
        latency = end - start
        measurements.append(
            {
                "latency": latency,
                "tokens": total_tokens,
                "tps": total_tokens / latency,
            }
        )

    result = summarize_measurements(measurements)
    del model, outputs, input_ids, attention_mask
    cleanup_cuda()
    return result


def print_results(results):
    print("\n=== Benchmark Results ===")
    for name, metrics in results.items():
        print(f"{name}:")
        for key, value in metrics.items():
            if key in {"total_tokens", "repeat"} and isinstance(value, (int, float)):
                print(f"  {key}: {int(round(value))}")
            elif key == "avg_tokens" and isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f} tokens/run")
            elif key in {"total_latency", "avg_latency"} and isinstance(
                value, (int, float)
            ):
                print(f"  {key}: {value:.4f} s")
            elif key == "tps" and isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f} tokens/s")
            elif isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark mini-vllm, vLLM, or transformers separately."
    )
    parser.add_argument(
        "--backend",
        choices=["minivllm", "vllm", "transformers"],
        required=True,
        help="Which single backend to benchmark.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=MAX_INPUT_TOKENS,
        help=(
            "Maximum input length and the fixed input length when "
            "--random-length is disabled."
        ),
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=NUM_SEQUENCES,
        help="Number of random input sequences in each generation batch.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help=(
            "Maximum number of new tokens and the fixed output limit when "
            "--random-length is disabled."
        ),
    )
    parser.add_argument(
        "--random-length",
        action="store_true",
        help=(
            "For mini-vLLM and vLLM, sample each request's input length and "
            "max_tokens from one eighth of the configured maximum through "
            "the maximum. "
            "Ignored for transformers."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=WARMUP_STEPS,
        help="Number of warmup generations before timing.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=REPEAT_STEPS,
        help="Number of timed generations to average.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Base random seed used before each timed generation.",
    )
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help=(
            "Stop at EOS instead of forcing each request toward its configured "
            "output token limit."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization passed to mini-vLLM and vLLM.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable mini-vLLM CUDA Graph capture and run eager decode.",
    )
    parser.add_argument(
        "--model-dtype",
        choices=["float32", "float16", "bfloat16"],
        default=MODEL_DTYPE,
        help="Model dtype used by mini-vLLM, vLLM, and transformers backends.",
    )
    args = parser.parse_args()
    if args.max_input_tokens <= 0:
        parser.error("--max-input-tokens must be greater than 0")
    if args.num_sequences <= 0:
        parser.error("--num-sequences must be greater than 0")
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be greater than 0")
    if (
        args.max_input_tokens + args.max_output_tokens
        > MODEL_CONFIG["max_position_embeddings"]
    ):
        parser.error(
            "--max-input-tokens + --max-output-tokens must not exceed "
            f"{MODEL_CONFIG['max_position_embeddings']}"
        )
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be at least 0")
    if args.repeat <= 0:
        parser.error("--repeat must be greater than 0")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


def main():
    global MAX_INPUT_TOKENS, NUM_SEQUENCES, MAX_OUTPUT_TOKENS
    global WARMUP_STEPS, REPEAT_STEPS, SEED, IGNORE_EOS
    global ENFORCE_EAGER, MODEL_DTYPE
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_tps.py requires a CUDA-capable GPU")
    MAX_INPUT_TOKENS = args.max_input_tokens
    NUM_SEQUENCES = args.num_sequences
    MAX_OUTPUT_TOKENS = args.max_output_tokens
    WARMUP_STEPS = args.warmup_steps
    REPEAT_STEPS = args.repeat
    SEED = args.seed
    IGNORE_EOS = not args.respect_eos
    ENFORCE_EAGER = args.enforce_eager
    MODEL_DTYPE = args.model_dtype
    set_seed(SEED)

    random_length_enabled = args.random_length and args.backend in {
        "minivllm",
        "vllm",
    }
    if args.random_length and not random_length_enabled:
        print("--random-length is ignored for the transformers backend.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Keep warmup and timed prompts distinct so prefix caching cannot inflate TPS.
    prompt_batches, output_token_limit_batches = build_random_token_workload_batches(
        tokenizer,
        MAX_INPUT_TOKENS,
        MAX_OUTPUT_TOKENS,
        NUM_SEQUENCES,
        WARMUP_STEPS + REPEAT_STEPS,
        SEED,
        random_length_enabled,
    )

    input_lengths = [
        len(prompt)
        for prompt_batch in prompt_batches[WARMUP_STEPS:]
        for prompt in prompt_batch
    ]
    output_token_limits = [
        output_token_limit
        for output_token_limit_batch in output_token_limit_batches[WARMUP_STEPS:]
        for output_token_limit in output_token_limit_batch
    ]
    length_mode = "random" if random_length_enabled else "fixed"
    configured_min_input_tokens = (
        (MAX_INPUT_TOKENS + 7) // 8
        if random_length_enabled
        else MAX_INPUT_TOKENS
    )
    configured_min_output_tokens = (
        (MAX_OUTPUT_TOKENS + 7) // 8
        if random_length_enabled
        else MAX_OUTPUT_TOKENS
    )
    print(
        f"Benchmark lengths ({length_mode}): {NUM_SEQUENCES} sequences/run; "
        f"input range [{configured_min_input_tokens}, {MAX_INPUT_TOKENS}] "
        f"(actual min/avg/max: {min(input_lengths)}/"
        f"{sum(input_lengths) / len(input_lengths):.2f}/{max(input_lengths)}); "
        f"sampling max_tokens range "
        f"[{configured_min_output_tokens}, {MAX_OUTPUT_TOKENS}] "
        f"(actual min/avg/max: {min(output_token_limits)}/"
        f"{sum(output_token_limits) / len(output_token_limits):.2f}/"
        f"{max(output_token_limits)})"
    )

    results = {}

    if args.backend == "minivllm":
        print("Running minivllm benchmark...")
        results["minivllm"] = run_minivllm(
            prompt_batches,
            output_token_limit_batches,
            random_length_enabled,
            args.gpu_memory_utilization,
        )

    if args.backend == "vllm":
        print("Running vLLM benchmark...")
        results["vLLM"] = run_vllm(
            prompt_batches,
            output_token_limit_batches,
            random_length_enabled,
            args.gpu_memory_utilization,
        )

    if args.backend == "transformers":
        print("Running transformers benchmark...")
        results["transformers"] = run_transformers_test(
            tokenizer, prompt_batches
        )

    print_results(results)


if __name__ == "__main__":
    main()
