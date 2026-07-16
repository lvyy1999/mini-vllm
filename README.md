# Mini-vLLM

基于 Nano-vLLM 扩展的轻量级 LLM 推理引擎。项目实现了 Triton prefill FlashAttention 和 decode PagedAttention，并包含 Paged KV cache、前缀缓存、chunked prefill、decode CUDA Graph 和张量并行等推理机制。

当前支持以下模型：

- `Qwen/Qwen3-0.6B`
- `meta-llama/Llama-3.2-1B-Instruct`

更完整的实现说明见[学习文档](docs/learn.md)。

## 快速开始

项目依赖 Linux、支持 CUDA 的 NVIDIA GPU，以及与 CUDA 环境匹配的 PyTorch。先安装 PyTorch，再安装其余运行依赖：

```bash
git clone https://github.com/lvyy1999/mini-vllm
cd mini-vllm

python3 -m pip install transformers==4.51.1 safetensors huggingface-hub triton tqdm numpy xxhash packaging
python3 main.py
```

首次运行会从 Hugging Face 下载 Qwen3-0.6B 的 tokenizer、配置和 safetensors 权重，因此需要能够访问 Hugging Face。离线运行时，可将入口脚本中的 `model` 改为完整的本地模型目录；当前模型工厂按目录名识别模型，因此目录名需要保留为 `Qwen3-0.6B` 或 `Llama-3.2-1B-Instruct`。运行 `benchmark_tps.py --backend vllm` 还需要单独安装 vLLM；下文结果使用的版本是 `vllm==0.8.5`。

`main.py` 演示了使用自定义引擎实现的完整 LLM 推理流程：

- 基于 Qwen3-0.6B，从 Hugging Face 下载权重并加载
- 创建多个聊天 prompt
- 通过自定义 LLM 引擎批处理 prompt
- 使用 PagedAttention 和 KV cache 管理来提高推理效率
- 每个 prompt 生成最多 256 个 tokens，采用温度采样

## 项目结构

```
mini-vllm/
├── src/minivllm/
│   ├── models/                 # Qwen3 和 Llama 3.2 模型实现
│   ├── engine/                 # 调度器、KV cache、执行引擎和 ModelRunner
│   ├── layers/                 # Triton attention 与模型组件
│   ├── utils/                  # 配置、运行上下文和权重加载
│   ├── llm.py                  # 顶层 LLM 接口
│   └── sampling_parameters.py  # 采样参数
├── docs/learn.md               # 实现原理与代码导读
├── results/                    # Benchmark 原始命令与输出
├── tests/                      # Scheduler 测试
├── main.py                     # Qwen3-0.6B 推理演示
├── main_llama32.py             # Llama-3.2-1B-Instruct 推理演示
├── benchmark_prefilling.py     # Prefill attention 微基准
├── benchmark_decoding.py       # Decode attention 微基准
└── benchmark_tps.py            # 端到端输出 TPS benchmark
```

## Benchmark 测试结果

测试环境：

| 项目 | 配置 |
|---|---|
| 平台 | AutoDL，单 GPU |
| GPU | NVIDIA A100 PCIe 40GB |
| CPU | 型号未记录，PyTorch 使用 10 个线程 |
| PyTorch | 2.6.0+cu124 |
| PyTorch CUDA build | 12.4 |
| vLLM | 0.8.5 |
| Transformers | 4.51.1 |
| 端到端模型 | Qwen/Qwen3-0.6B |
| 端到端模型 dtype | BF16 |

Attention 微基准使用合成输入，配置为 32 个 query heads、8 个 KV heads、`head_dim=128`；decode 测试的 KV cache block size 为 16。CPU PyTorch 使用 FP32 和 10 个 CPU 线程。GPU PyTorch 从 FP16 输入开始，但参考实现会将 Q/K/V 转成 FP32，并包含 Python 循环、临时 tensor 分配以及 prefill 的 causal mask 构造或 decode 的分页 K/V 重建。Triton 路径使用 FP16 输入。计时采用 wall-clock time，并在 GPU 计时边界执行 CUDA synchronize。输入生成和 CPU/GPU 数据传输不计时，但单次函数调用内部的分配、wrapper 和 kernel launch 开销包含在结果中。

因此，GPU PyTorch 数据应理解为用于正确性验证的朴素参考实现，而不是 PyTorch SDPA、FlashAttention 或其他融合 attention kernel。CPU/GPU 和 Triton/GPU PyTorch 加速比只适用于这里定义的实现和输入形状，不能直接代表相对生产级 attention backend 的通用加速比。由于测试时没有记录 CPU 型号，CPU 绝对延迟及其相关加速比也不能用于跨机器复现。

### Prefill Attention 测试

运行命令：

```bash
python3 benchmark_prefilling.py
```

[完整输出](results/benchmark_prefilling.txt)

| Batch / 序列长度 | 总 token 数 | CPU PyTorch FP32 | GPU PyTorch（FP16 输入，FP32 计算） | GPU Triton（FP16 输入） | Triton 相对 CPU | Triton 相对 GPU PyTorch | Triton vs CPU 最大绝对误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 x 60 | 120 | 1.046 ms | 0.620 ms | 0.050 ms | 20.79x | 12.31x | 0.001848 |
| 4 x 64 | 256 | 2.651 ms | 1.236 ms | 0.044 ms | 60.65x | 28.26x | 0.001878 |
| 2 x 1024 | 2048 | 169.701 ms | 5.760 ms | 0.503 ms | 337.32x | 11.45x | 0.001672 |
| 1 x 4096 | 4096 | 2677.379 ms | 28.361 ms | 3.729 ms | 718.06x | 7.61x | 0.001583 |

CPU PyTorch FP32 输出作为正确性参考，判定条件为 `torch.allclose(atol=0.02, rtol=0.02)`。GPU PyTorch 和 Triton 在全部形状上均通过，所有已报告的最大绝对误差不超过 `0.001953`。在当前朴素 GPU PyTorch 参考实现下，Triton prefill 调用获得 **7.61x-28.26x** 的加速；延迟从短输入的约 `0.05 ms` 增长到 4096-token 输入的 `3.729 ms`，变化趋势与 attention 工作量增长一致。

### Decode PagedAttention 测试

运行命令：

```bash
python3 benchmark_decoding.py
```

[完整输出](results/benchmark_decoding.txt)

| Batch 大小 | 上下文长度 | CPU PyTorch FP32 | GPU PyTorch（FP16 输入，FP32 计算） | GPU Triton（FP16 输入） | Triton 相对 CPU | Triton 相对 GPU PyTorch | Triton vs CPU 最大绝对误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 60 | 1.193 ms | 0.649 ms | 0.069 ms | 17.18x | 9.34x | 0.001494 |
| 1 | 512 | 2.743 ms | 0.805 ms | 0.074 ms | 37.22x | 10.91x | 0.000364 |
| 16 | 256 | 54.722 ms | 2.344 ms | 0.043 ms | 1263.40x | 54.12x | 0.001570 |
| 4 | 2048 | 145.562 ms | 1.901 ms | 0.342 ms | 426.16x | 5.57x | 0.000228 |

GPU PyTorch 和 Triton 在全部形状上均通过正确性检查，最大绝对误差不超过 `0.001570`。Triton kernel 通过 block table 直接读取分页 KV cache，而 PyTorch 参考实现会重建、padding 并扩展 K/V，因此这里的 **5.57x-54.12x** 加速同时包含了避免这些额外操作的收益。几十微秒级结果也容易受 GPU 频率、launch 开销和计时方法影响，表中的时间应视为完整算子调用的平均延迟，而不是隔离后的 PTX kernel 延迟。

### 端到端输出 TPS 测试

端到端 benchmark 使用随机采样的非特殊 token ID，并分别在独立进程中运行 mini-vLLM、vLLM V0、vLLM V1 和 Transformers。模型加载、tokenizer 加载、workload 构造和 warmup 不计时；prefill、逐 token decode、调度和采样均计入。测试未传入 `--enforce-eager`，因此 mini-vLLM 和 vLLM 均以 `enforce_eager=False` 运行，允许使用 CUDA Graph。默认忽略 EOS，每个 warmup 和计时 repeat 使用不同 prompt，避免重复 prompt 带来的前缀缓存命中。

mini-vLLM 和 vLLM 都会在计时区间内将 completion token IDs 解码为文本：mini-vLLM 的 `generate()` 显式调用 tokenizer，vLLM 未设置 `detokenize`，使用默认值 `True`。Transformers 的 `generate()` 只返回 token tensor，因此 Transformers 结果不包含文本解码开销。mini-vLLM 与 vLLM 的功能口径已经对齐，但如果要进行严格的纯生成吞吐量对比，仍应让所有后端统一关闭 detokenization；如果要比较面向用户的完整生成流程，则应为 Transformers 补上计时区间内的文本解码。

TPS 的统计口径为：

```text
output TPS = 所有计时 repeat 的实际生成 token 总数 / 总生成耗时
```

输入 token 会增加耗时，但不计入 TPS 分子。

#### 固定长度

每个 repeat 包含 64 条请求，每条请求固定输入 512 tokens，并固定生成 512 tokens。完整命令：

```bash
python3 benchmark_tps.py \
  --backend minivllm \
  --max-input-tokens 512 \
  --num-sequences 64 \
  --max-output-tokens 512 \
  --warmup-steps 1 \
  --repeat 3 \
  --seed 0 \
  --model-dtype bfloat16 \
  --gpu-memory-utilization 0.9

VLLM_USE_V1=0 python3 benchmark_tps.py \
  --backend vllm \
  --max-input-tokens 512 \
  --num-sequences 64 \
  --max-output-tokens 512 \
  --warmup-steps 1 \
  --repeat 3 \
  --seed 0 \
  --model-dtype bfloat16 \
  --gpu-memory-utilization 0.9

VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 benchmark_tps.py \
  --backend vllm \
  --max-input-tokens 512 \
  --num-sequences 64 \
  --max-output-tokens 512 \
  --warmup-steps 1 \
  --repeat 3 \
  --seed 0 \
  --model-dtype bfloat16 \
  --gpu-memory-utilization 0.9

python3 benchmark_tps.py \
  --backend transformers \
  --max-input-tokens 512 \
  --num-sequences 64 \
  --max-output-tokens 512 \
  --warmup-steps 1 \
  --repeat 3 \
  --seed 0 \
  --model-dtype bfloat16
```

| 推理后端 | 引擎模式 | 总耗时 | 总输出 token 数 | 单次平均耗时 | 单次平均输出 token 数 | TPS |
|---|---|---:|---:|---:|---:|---:|
| mini-vLLM | CUDA Graph | 18.7222 s | 98304 | 6.2407 s | 32768.00 | 5250.6584 tokens/s |
| vLLM | V0 | 20.3883 s | 98304 | 6.7961 s | 32768.00 | 4821.5941 tokens/s |
| vLLM | V1 | 15.5153 s | 98304 | 5.1718 s | 32768.00 | 6335.9521 tokens/s |
| Transformers | generate | 101.1002 s | 98304 | 33.7001 s | 32768.00 | 972.3426 tokens/s |

在该固定长度 workload 下，vLLM V1 吞吐量最高。mini-vLLM 达到 vLLM V1 的 **82.9%**，是 vLLM V0 的 **1.09x**、Transformers 的 **5.40x**；相对 vLLM V0 高 **8.9%**。该结果说明 mini-vLLM 在这一规则的大 batch workload 下接近 vLLM V0，但仍落后于 vLLM V1。由于没有提供 `--enforce-eager` 对照，当前结果不能单独量化 CUDA Graph 的贡献，也不能将差距归因于某一项具体机制。

#### 随机长度

启用 `--random-length` 后，每条请求的输入长度和 `max_tokens` 分别在 `[128, 1024]` 中独立均匀采样。Transformers backend 当前忽略该选项，因此随机长度结果只比较 mini-vLLM、vLLM V0 和 vLLM V1。完整命令：

```bash
python3 benchmark_tps.py \
  --backend minivllm \
  --max-input-tokens 1024 \
  --num-sequences 64 \
  --max-output-tokens 1024 \
  --warmup-steps 1 \
  --repeat 3 \
  --seed 0 \
  --model-dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --random-length

VLLM_USE_V1=0 python3 benchmark_tps.py \
  --backend vllm \
  --max-input-tokens 1024 \
  --num-sequences 64 \
  --max-output-tokens 1024 \
  --warmup-steps 1 \
  --repeat 3 \
  --seed 0 \
  --model-dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --random-length

VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 benchmark_tps.py \
  --backend vllm \
  --max-input-tokens 1024 \
  --num-sequences 64 \
  --max-output-tokens 1024 \
  --warmup-steps 1 \
  --repeat 3 \
  --seed 0 \
  --model-dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --random-length
```

| 推理后端 | 引擎模式 | 总耗时 | 总输出 token 数 | 单次平均耗时 | 单次平均输出 token 数 | TPS |
|---|---|---:|---:|---:|---:|---:|
| mini-vLLM | CUDA Graph | 33.9433 s | 112219 | 11.3144 s | 37406.33 | 3306.0748 tokens/s |
| vLLM | V0 | 32.1858 s | 112219 | 10.7286 s | 37406.33 | 3486.6007 tokens/s |
| vLLM | V1 | 26.4411 s | 112219 | 8.8137 s | 37406.33 | 4244.1133 tokens/s |

三种引擎生成的 token 总数完全一致，说明逐请求 `max_tokens` workload 已正确对齐。在该随机长度 workload 下，mini-vLLM 达到 vLLM V1 的 **77.9%**，并达到 vLLM V0 的 **94.8%**；vLLM V0 比 mini-vLLM 高 **5.5%**。随机输入和输出长度会使活跃 batch size 持续变化，可能影响 CUDA Graph batch 对齐、调度和 KV cache 管理，但当前测试不能分离这些因素各自的影响。

固定长度与随机长度测试使用了不同的最大输入/输出长度，不能直接用两张表判断随机化本身带来的性能变化。两组测试均只进行了 1 次 warmup 和 3 次计时，没有报告方差。当前数据适合作为各自 workload 下的初步实测记录；若要形成更稳定的结论，应补充相同最大长度下的固定/随机对照，增加 warmup、repeat 和独立重复运行，并报告离散程度。

[端到端完整命令与输出](results/benchmark_tps.txt)
