import sys
from pathlib import Path
import time

import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from minivllm.layers import flash_attention_decode


def report_correctness(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> bool:
    reference_cpu = reference.detach().to(device="cpu", dtype=torch.float32)
    candidate_cpu = candidate.detach().to(device="cpu", dtype=torch.float32)
    max_abs_err = (reference_cpu - candidate_cpu).abs().max().item()
    is_close = torch.allclose(reference_cpu, candidate_cpu, atol=atol, rtol=rtol)
    status = "PASS" if is_close else "FAIL"
    print(f"   Correctness vs CPU PyTorch [{name}]: {status}, max_abs_err={max_abs_err:.6f}")
    return is_close


def decode_pytorch_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """Paged-KV decode baseline that runs on the input tensors' device."""
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype
    max_context_len = context_lens.max().item()

    padded_k = torch.zeros(
        batch_size, max_context_len, num_kv_heads, head_dim,
        device=device, dtype=dtype,
    )
    padded_v = torch.zeros_like(padded_k)

    for i in range(batch_size):
        seq_len = context_lens[i].item()
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        valid_blocks = block_tables[i, :num_blocks_needed]
        valid_blocks = valid_blocks[valid_blocks != -1]

        if len(valid_blocks) > 0:
            gathered_k = k_cache[valid_blocks].reshape(
                -1, num_kv_heads, head_dim
            )[:seq_len]
            gathered_v = v_cache[valid_blocks].reshape(
                -1, num_kv_heads, head_dim
            )[:seq_len]
            padded_k[i, :seq_len] = gathered_k
            padded_v[i, :seq_len] = gathered_v

    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        padded_k = padded_k.repeat_interleave(num_groups, dim=2)
        padded_v = padded_v.repeat_interleave(num_groups, dim=2)

    q = q.unsqueeze(2)
    padded_k = padded_k.transpose(1, 2)
    padded_v = padded_v.transpose(1, 2)

    attn_scores = torch.matmul(q, padded_k.transpose(-2, -1)) * scale
    mask = torch.arange(max_context_len, device=device)[None, :] < context_lens[:, None]
    attn_scores = attn_scores.masked_fill(
        ~mask[:, None, None, :], float("-inf")
    )
    attn_probs = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, padded_v).squeeze(2)


def setup_test_data(batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size):
    max_num_blocks = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * max_num_blocks

    q_cpu = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float32)
    k_cache_cpu = torch.randn(
        total_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float32
    )
    v_cache_cpu = torch.randn_like(k_cache_cpu)
    block_tables_cpu = torch.arange(total_blocks, dtype=torch.int32).reshape(
        batch_size, max_num_blocks
    )
    context_lens_cpu = torch.full((batch_size,), seq_len, dtype=torch.int32)

    q_gpu = q_cpu.to(device="cuda", dtype=torch.float16)
    k_cache_gpu = k_cache_cpu.to(device="cuda", dtype=torch.float16)
    v_cache_gpu = v_cache_cpu.to(device="cuda", dtype=torch.float16)
    block_tables_gpu = block_tables_cpu.to(device="cuda")
    context_lens_gpu = context_lens_cpu.to(device="cuda")

    scale = 1.0 / (head_dim ** 0.5)
    cpu_inputs = (
        q_cpu, k_cache_cpu, v_cache_cpu, block_tables_cpu, context_lens_cpu,
    )
    gpu_inputs = (
        q_gpu, k_cache_gpu, v_cache_gpu, block_tables_gpu, context_lens_gpu,
    )
    return cpu_inputs, gpu_inputs, scale


def cpu_iteration_count(batch_size: int, seq_len: int, requested: int) -> int:
    if batch_size * seq_len <= 4096:
        return min(requested, 10)
    return min(requested, 3)


def benchmark(
    batch_size,
    seq_len,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    block_size=16,
    num_iterations=100,
):
    print(f"\n{'=' * 70}")
    print(f"batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}")
    print(f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, block_size={block_size}")
    print(f"{'=' * 70}")

    cpu_inputs, gpu_inputs, scale = setup_test_data(
        batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size
    )
    q_cpu, k_cpu, v_cpu, blocks_cpu, lens_cpu = cpu_inputs
    q_gpu, k_gpu, v_gpu, blocks_gpu, lens_gpu = gpu_inputs
    cpu_iters = cpu_iteration_count(batch_size, seq_len, num_iterations)
    results = {}

    print(f"\n[1/3] CPU PyTorch baseline (FP32, {cpu_iters} iterations)...")
    for _ in range(2):
        _ = decode_pytorch_attention(
            q_cpu, k_cpu, v_cpu, blocks_cpu, lens_cpu, scale,
            num_heads, num_kv_heads, head_dim, block_size,
        )
    start = time.perf_counter()
    for _ in range(cpu_iters):
        out_cpu = decode_pytorch_attention(
            q_cpu, k_cpu, v_cpu, blocks_cpu, lens_cpu, scale,
            num_heads, num_kv_heads, head_dim, block_size,
        )
    cpu_time = (time.perf_counter() - start) / cpu_iters
    results["CPU PyTorch FP32"] = cpu_time
    print(f"   Time: {cpu_time * 1000:.3f} ms")

    print(f"\n[2/3] GPU PyTorch baseline (FP16, {num_iterations} iterations)...")
    for _ in range(10):
        _ = decode_pytorch_attention(
            q_gpu, k_gpu, v_gpu, blocks_gpu, lens_gpu, scale,
            num_heads, num_kv_heads, head_dim, block_size,
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out_gpu = decode_pytorch_attention(
            q_gpu, k_gpu, v_gpu, blocks_gpu, lens_gpu, scale,
            num_heads, num_kv_heads, head_dim, block_size,
        )
    torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - start) / num_iterations
    results["GPU PyTorch FP16"] = gpu_time
    print(f"   Time: {gpu_time * 1000:.3f} ms")
    report_correctness("GPU PyTorch FP16", out_cpu, out_gpu)

    print(f"\n[3/3] GPU Triton PagedAttention (FP16, {num_iterations} iterations)...")
    for _ in range(10):
        _ = flash_attention_decode(q_gpu, k_gpu, v_gpu, scale, lens_gpu, blocks_gpu)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out_triton = flash_attention_decode(
            q_gpu, k_gpu, v_gpu, scale, lens_gpu, blocks_gpu
        )
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / num_iterations
    results["GPU Triton FP16"] = triton_time
    print(f"   Time: {triton_time * 1000:.3f} ms")
    report_correctness("GPU Triton FP16", out_cpu, out_triton)

    print("\n   Speedups:")
    print(f"   GPU PyTorch vs CPU PyTorch: {cpu_time / gpu_time:.2f}x")
    print(f"   GPU Triton vs CPU PyTorch:  {cpu_time / triton_time:.2f}x")
    print(f"   GPU Triton vs GPU PyTorch:  {gpu_time / triton_time:.2f}x")
    return results


if __name__ == "__main__":
    torch.manual_seed(0)

    print("\n" + "=" * 70)
    print("PAGED ATTENTION DECODE BENCHMARK")
    print("Comparing: CPU PyTorch FP32 | GPU PyTorch FP16 | GPU Triton FP16")
    print(f"CPU threads: {torch.get_num_threads()}")
    print("=" * 70)

    benchmark(batch_size=2, seq_len=60, num_iterations=100)
    benchmark(batch_size=1, seq_len=512, num_iterations=100)
    benchmark(batch_size=16, seq_len=256, num_iterations=50)
    benchmark(batch_size=4, seq_len=2048, num_iterations=20)
