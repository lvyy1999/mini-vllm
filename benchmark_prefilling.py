import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from minivllm.layers import flash_attention_prefill


def report_correctness(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 2e-2,
    rtol: float = 2e-2,
    reference_name: str = "CPU PyTorch",
) -> bool:
    reference_cpu = reference.detach().to(device="cpu", dtype=torch.float32)
    candidate_cpu = candidate.detach().to(device="cpu", dtype=torch.float32)
    max_abs_err = (reference_cpu - candidate_cpu).abs().max().item()
    is_close = torch.allclose(reference_cpu, candidate_cpu, atol=atol, rtol=rtol)
    status = "PASS" if is_close else "FAIL"
    print(
        f"      Correctness vs {reference_name} [{name}]: {status}, max_abs_err={max_abs_err:.6f}"
    )
    return is_close


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_prefilling.py requires a CUDA-capable GPU")


def assert_correctness(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 2e-2,
    rtol: float = 2e-2,
    reference_name: str = "CPU PyTorch",
) -> None:
    if not report_correctness(name, reference, candidate, atol, rtol, reference_name):
        raise AssertionError(f"{name} does not match {reference_name}")


def pytorch_standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Standard causal attention that runs on the input tensors' device."""
    output = torch.zeros_like(q)
    cu_seqlens_cpu = cu_seqlens.cpu().tolist()

    for i in range(len(cu_seqlens_cpu) - 1):
        start = cu_seqlens_cpu[i]
        end = cu_seqlens_cpu[i + 1]
        seq_len = end - start

        q_seq = q[start:end].transpose(0, 1)
        k_seq = k[start:end].transpose(0, 1)
        v_seq = v[start:end].transpose(0, 1)

        if num_kv_heads != num_heads:
            num_groups = num_heads // num_kv_heads
            k_seq = k_seq.repeat_interleave(num_groups, dim=0)
            v_seq = v_seq.repeat_interleave(num_groups, dim=0)

        attn_scores = torch.matmul(q_seq, k_seq.transpose(1, 2)) * scale
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output[start:end] = torch.matmul(attn_probs, v_seq).transpose(0, 1)

    return output


def setup_data(num_seqs, seq_len, num_heads, num_kv_heads, head_dim):
    total_tokens = num_seqs * seq_len

    q_cpu = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float32)
    k_cpu = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float32)
    v_cpu = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float32)
    cu_seqlens_cpu = torch.tensor(
        [i * seq_len for i in range(num_seqs + 1)],
        dtype=torch.int32,
    )

    q_gpu = q_cpu.to(device="cuda", dtype=torch.float16)
    k_gpu = k_cpu.to(device="cuda", dtype=torch.float16)
    v_gpu = v_cpu.to(device="cuda", dtype=torch.float16)
    cu_seqlens_gpu = cu_seqlens_cpu.to(device="cuda")

    scale = 1.0 / (head_dim**0.5)
    cpu_inputs = (q_cpu, k_cpu, v_cpu, cu_seqlens_cpu)
    gpu_inputs = (q_gpu, k_gpu, v_gpu, cu_seqlens_gpu)
    return cpu_inputs, gpu_inputs, scale


def cpu_iteration_count(seq_len: int, requested: int) -> int:
    if seq_len <= 256:
        return min(requested, 10)
    if seq_len <= 1024:
        return min(requested, 3)
    return 1


def benchmark(
    num_seqs, seq_len, num_heads=32, num_kv_heads=8, head_dim=128, num_iter=50
):
    print(f"\n{'=' * 80}")
    print(
        f"Benchmark: {num_seqs} seqs x {seq_len} tokens (total: {num_seqs * seq_len} tokens)"
    )
    print(f"Heads: {num_heads}/{num_kv_heads}, Dim: {head_dim}")
    print(f"{'=' * 80}")

    cpu_inputs, gpu_inputs, scale = setup_data(
        num_seqs, seq_len, num_heads, num_kv_heads, head_dim
    )
    q_cpu, k_cpu, v_cpu, cu_seqlens_cpu = cpu_inputs
    q_gpu, k_gpu, v_gpu, cu_seqlens_gpu = gpu_inputs
    cpu_iters = cpu_iteration_count(seq_len, num_iter)
    results = {}

    print(f"\n[1/3] CPU PyTorch baseline (FP32, {cpu_iters} iterations)...")
    if seq_len <= 1024:
        _ = pytorch_standard_attention(
            q_cpu,
            k_cpu,
            v_cpu,
            cu_seqlens_cpu,
            scale,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    start = time.perf_counter()
    for _ in range(cpu_iters):
        out_cpu = pytorch_standard_attention(
            q_cpu,
            k_cpu,
            v_cpu,
            cu_seqlens_cpu,
            scale,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    cpu_time = (time.perf_counter() - start) / cpu_iters
    results["CPU PyTorch FP32"] = cpu_time
    print(f"      Time: {cpu_time * 1000:.3f} ms")

    print(f"\n[2/3] GPU PyTorch baseline (FP16, {num_iter} iterations)...")
    for _ in range(5):
        _ = pytorch_standard_attention(
            q_gpu,
            k_gpu,
            v_gpu,
            cu_seqlens_cpu,
            scale,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        out_gpu = pytorch_standard_attention(
            q_gpu,
            k_gpu,
            v_gpu,
            cu_seqlens_cpu,
            scale,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - start) / num_iter
    results["GPU PyTorch FP16"] = gpu_time
    print(f"      Time: {gpu_time * 1000:.3f} ms")
    assert_correctness("GPU PyTorch FP16", out_cpu, out_gpu)

    print(f"\n[3/3] GPU Triton Flash Attention (FP16, {num_iter} iterations)...")
    for _ in range(5):
        _ = flash_attention_prefill(
            q_gpu,
            k_gpu,
            v_gpu,
            scale,
            cu_seqlens_gpu,
            max_seqlen_q=seq_len,
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        out_triton = flash_attention_prefill(
            q_gpu,
            k_gpu,
            v_gpu,
            scale,
            cu_seqlens_gpu,
            max_seqlen_q=seq_len,
        )
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / num_iter
    results["GPU Triton FP16"] = triton_time
    print(f"      Time: {triton_time * 1000:.3f} ms")
    assert_correctness("GPU Triton FP16", out_cpu, out_triton)
    assert_correctness(
        "GPU Triton FP16",
        out_gpu,
        out_triton,
        reference_name="GPU PyTorch FP16",
    )

    print("\n      Speedups:")
    print(f"      GPU PyTorch vs CPU PyTorch: {cpu_time / gpu_time:.2f}x")
    print(f"      GPU Triton vs CPU PyTorch:  {cpu_time / triton_time:.2f}x")
    print(f"      GPU Triton vs GPU PyTorch:  {gpu_time / triton_time:.2f}x")
    return results


if __name__ == "__main__":
    require_cuda()
    torch.manual_seed(0)

    print("\n" + "=" * 80)
    print("PREFILL ATTENTION BENCHMARK")
    print("Comparing: CPU PyTorch FP32 | GPU PyTorch FP16 | GPU Triton FP16")
    print(f"CPU threads: {torch.get_num_threads()}")
    print("=" * 80)

    benchmark(num_seqs=2, seq_len=60, num_iter=100)
    benchmark(num_seqs=4, seq_len=64, num_iter=100)
    benchmark(num_seqs=2, seq_len=1024, num_iter=30)
    benchmark(num_seqs=1, seq_len=4096, num_iter=10)
