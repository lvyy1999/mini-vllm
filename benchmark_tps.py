import time
import torch

from transformers import AutoTokenizer,AutoModelForCausalLM

# ===== minivllm =====
from minivllm.llm import LLM as MiniVLLM
from minivllm.sampling_parameters import SamplingParams as MiniSamplingParams

# ===== vllm =====
from vllm import LLM as VLLM
from vllm import SamplingParams as VLLMSamplingParams

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_CONFIG = {
  "architectures": [
    "Qwen3ForCausalLM"
  ],
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
  "vocab_size": 151936
}

PROMPTS = [
    "introduce yourself" ,
    "list all prime numbers within 100" ,
    "give me your opinion on the impact of artificial intelligence on society" ,
]

WARMUP_STEPS = 2
OUTPUT_TOKENS = 256  # ouput token num
device = "cuda" if torch.cuda.is_available() else "cpu"

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def run_minivllm(tokenizer):
    llm = MiniVLLM(enforce_eager=True, model_name_or_path=MODEL_NAME, custom_model_config=MODEL_CONFIG)
    sampling = MiniSamplingParams(
        temperature=0.6,
        max_tokens=OUTPUT_TOKENS,
    )

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    # warmup
    for _ in range(WARMUP_STEPS):
        llm.generate(prompts, sampling)
        cuda_sync()

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    cuda_sync()
    end = time.perf_counter()

    total_tokens = sum(len(output['token_ids']) for output in outputs)
    latency = end - start

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": total_tokens / latency,
    }


def run_vllm(tokenizer):
    # vLLM
    llm = VLLM(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        trust_remote_code=False, 
        gpu_memory_utilization=0.75,  
        max_model_len=256, 
        speculative_config=None, 
    )

    sampling = VLLMSamplingParams(
        temperature=0.6,
        max_tokens=OUTPUT_TOKENS,
    )

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    # warmup
    for _ in range(WARMUP_STEPS):
        llm.generate(prompts, sampling)
        cuda_sync()

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    cuda_sync()
    end = time.perf_counter()

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    latency = end - start

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": total_tokens / latency,
    }


def run_transformers_test(tokenizer):
    # transformers
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(device)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    # Prepare attention_mask explicitly
    attention_mask = inputs["attention_mask"]

    # warmup
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=OUTPUT_TOKENS)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=OUTPUT_TOKENS)
    end = time.perf_counter()

    total_tokens = sum(len(output) for output in outputs)
    latency = end - start

    tps = total_tokens / latency

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": tps,
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left')

    print("Running minivllm benchmark...")
    mini = run_minivllm(tokenizer)

    print("Running vLLM benchmark...")
    vllm = run_vllm(tokenizer)

    print("Running transformers benchmark...")
    transformers = run_transformers_test(tokenizer)


    results = {
        "minivllm": mini,
        "vLLM": vllm,
        "transformers":transformers
    }

    print("\n=== Benchmark Results ===")
    for k, v in results.items():
        print(f"{k}:")
        for kk, vv in v.items():
            print(f"  {kk}: {vv:.4f}")



if __name__ == "__main__":
    main()
