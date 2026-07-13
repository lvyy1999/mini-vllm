import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from minivllm import LLM, SamplingParams

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# config of used custom model
model_config = {
    'vocab_size': 128256,
    'hidden_size': 2048,
    'head_dim': 64,
    'num_qo_heads': 32,
    'num_kv_heads': 8,
    'has_attn_bias': False,
    'rms_norm_epsilon': 1e-5,
    'rope_base': 500000,
    'max_position_embeddings': 32768,
    'intermediate_size': 8192,
    'ffn_bias': False,
    'num_layers': 16,
    'tie_word_embeddings': True,
    'eos': 128009,
}

def main():
    model = "meta-llama/Llama-3.2-1B-Instruct"
    path = os.path.expanduser("~/huggingface/Llama-3.2-1B-Instruct/")
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
        print(f"Prompt: {prompt}")
        print(f"Completion: {output['text']}")


if __name__ == "__main__":
    main()