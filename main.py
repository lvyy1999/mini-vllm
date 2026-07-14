import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from minivllm import LLM, SamplingParams

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# config of used custom model
model_config = {
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

def main():
    model = "Qwen/Qwen3-0.6B"
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=path)
    llm = LLM(enforce_eager=True, model_name_or_path=model, custom_model_config=model_config)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",# * 15,
        "list all prime numbers within 100",# * 15,
        "give me your opinion on the impact of artificial intelligence on society",# * 15,
    ] # * 30
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