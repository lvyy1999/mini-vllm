import os
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from minivllm import LLM, SamplingParams

# config of used custom model
model_config = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": [128001, 128008, 128009],
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "float16",  # Tesla T4 not support bfloat16, change to float16
    "transformers_version": "4.45.0.dev0",
    "use_cache": True,
    "vocab_size": 128256,
}


def main():
    model = "meta-llama/Llama-3.2-1B-Instruct"
    path = os.path.expanduser("~/huggingface/Llama-3.2-1B-Instruct/")
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=path)
    llm = LLM(
        enforce_eager=True, model_name_or_path=model, custom_model_config=model_config
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",  # * 15,
        "list all prime numbers within 100",  # * 15,
        "give me your opinion on the impact of artificial intelligence on society",  # * 15,
    ]  # * 30
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
