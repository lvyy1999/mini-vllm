# Mini-vLLM Benchmark 详细记录

本文记录项目的完整测试环境、运行命令、统计口径、结果表和限制说明。所有命令均从项目根目录执行。

[返回 README](../README.md)

## 测试环境

只有复现 vLLM 端到端对比时需要额外安装 vLLM；运行 `main.py` 和 `benchmarks/benchmark_int8.py` 均不依赖 vLLM。

| 项目 | 配置 |
|---|---|
| 平台 | AutoDL，单 GPU |
| GPU | NVIDIA A100 PCIe 40GB |
| CPU | Intel Xeon Processor (Skylake, IBRS)，PyTorch 使用 10 个线程 |
| PyTorch | 2.6.0+cu124 |
| PyTorch CUDA build | 12.4 |
| Triton | 3.1.0 |
| vLLM | 0.8.5 |
| Transformers | 4.51.1 |
| 端到端模型 | Qwen/Qwen3-0.6B |
| 端到端 TPS 基线模型 dtype | BF16 |
| INT8 Benchmark 模式 | FP16 模型/缓存、BF16 模型/缓存、BF16 模型 + INT8 KV Cache |

## 测试口径

Attention 微基准使用合成输入，配置为 32 个 query heads、8 个 KV heads、`head_dim=128`；Decode 测试的 KV Cache block size 为 16。CPU PyTorch 使用 FP32 和 10 个 CPU 线程。GPU PyTorch 从 FP16 输入开始，但参考实现会将 Q/K/V 转成 FP32，并包含 Python 循环、临时 tensor 分配，以及 Prefill 的 causal mask 构造或 Decode 的分页 K/V 重建。Triton 路径使用 FP16 输入。

计时采用 wall-clock time，并在 GPU 计时边界执行 CUDA synchronize。输入生成和 CPU/GPU 数据传输不计时，但单次函数调用内部的分配、wrapper 和 kernel launch 开销包含在结果中。

GPU PyTorch 数据应理解为用于数值验证的朴素参考实现，而不是 PyTorch SDPA、FlashAttention 或其他融合 Attention Kernel。CPU/GPU 和 Triton/GPU PyTorch 加速比只适用于本文定义的实现与输入形状，不能直接代表相对生产级 Attention Backend 的通用加速比。AutoDL 的虚拟化环境、CPU 频率和资源调度可能变化，因此 CPU 绝对延迟及相关加速比不宜直接用于跨机器对比。

## Prefill FlashAttention

当前脚本没有 CLI 参数，测试形状、head 数、dtype 和迭代次数均在脚本中固定配置。

运行命令：

```bash
python3 benchmarks/benchmark_prefilling.py
```

| Batch / 序列长度 | 总 token 数 | CPU PyTorch FP32 | GPU PyTorch（FP16 输入，FP32 计算） | GPU Triton（FP16 输入） | Triton 相对 CPU | Triton 相对 GPU PyTorch | Triton vs CPU 最大绝对误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 x 60 | 120 | 1.046 ms | 0.620 ms | 0.050 ms | 20.79x | 12.31x | 0.001848 |
| 4 x 64 | 256 | 2.651 ms | 1.236 ms | 0.044 ms | 60.65x | 28.26x | 0.001878 |
| 2 x 1024 | 2048 | 169.701 ms | 5.760 ms | 0.503 ms | 337.32x | 11.45x | 0.001672 |
| 1 x 4096 | 4096 | 2677.379 ms | 28.361 ms | 3.729 ms | 718.06x | 7.61x | 0.001583 |

CPU PyTorch FP32 输出作为数值参考，判定条件为 `torch.allclose(atol=0.02, rtol=0.02)`。GPU PyTorch 和 Triton 在全部形状上均通过，所有已报告的最大绝对误差不超过 `0.001953`。

在当前朴素 GPU PyTorch 参考实现下，Triton Prefill 调用获得 **7.61x-28.26x** 的加速；延迟从短输入的约 `0.05 ms` 增长到 4096-token 输入的 `3.729 ms`。该加速比不能直接用于对比生产级融合 Attention Kernel。

## Decode PagedAttention

当前脚本没有 CLI 参数，测试形状、KV Cache block size、dtype 和迭代次数均在脚本中固定配置。

运行命令：

```bash
python3 benchmarks/benchmark_decoding.py
```

| Batch 大小 | 上下文长度 | CPU PyTorch FP32 | GPU PyTorch（FP16 输入，FP32 计算） | GPU Triton（FP16 输入） | Triton 相对 CPU | Triton 相对 GPU PyTorch | Triton vs CPU 最大绝对误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 60 | 1.193 ms | 0.649 ms | 0.069 ms | 17.18x | 9.34x | 0.001494 |
| 1 | 512 | 2.743 ms | 0.805 ms | 0.074 ms | 37.22x | 10.91x | 0.000364 |
| 16 | 256 | 54.722 ms | 2.344 ms | 0.043 ms | 1263.40x | 54.12x | 0.001570 |
| 4 | 2048 | 145.562 ms | 1.901 ms | 0.342 ms | 426.16x | 5.57x | 0.000228 |

GPU PyTorch 和 Triton 在全部形状上均通过数值检查，最大绝对误差不超过 `0.001570`。Triton Kernel 通过 block table 直接读取分页 KV Cache，而 PyTorch 参考实现会重建、padding 并扩展 K/V，因此 **5.57x-54.12x** 的加速同时包含避免这些额外操作的收益。

几十微秒级结果容易受到 GPU 频率、launch 开销和计时方法影响，表中时间应视为完整算子调用的平均延迟，而不是隔离后的 PTX Kernel 延迟。

## 端到端输出 TPS

端到端 Benchmark 使用随机采样的非特殊 token ID，并分别在独立进程中运行 mini-vLLM、vLLM V0、vLLM V1 和 Transformers。模型加载、tokenizer 加载、workload 构造和 warmup 不计时；Prefill、逐 token Decode、调度和采样均计入。

测试未传入 `--enforce-eager`，因此 mini-vLLM 和 vLLM 均以 `enforce_eager=False` 运行，允许使用 CUDA Graph 优化解码。默认忽略 EOS，每个 warmup 和计时 repeat 使用不同 prompt，避免重复 prompt 带来的 Prefix Cache 命中。

mini-vLLM 和 vLLM 都会在计时区间内将 completion token IDs 解码为文本：mini-vLLM 的 `generate()` 显式调用 tokenizer，vLLM 未设置 `detokenize`，使用默认值 `True`。Transformers 的 `generate()` 只返回 token tensor，因此 Transformers 结果不包含文本解码开销。

mini-vLLM 与 vLLM 的功能口径已经对齐。如果要进行严格的纯生成吞吐对比，应让所有 Backend 统一关闭 detokenization；如果要比较面向用户的完整生成流程，则应为 Transformers 补上计时区间内的文本解码。

### TPS 参数

下表记录 `benchmarks/benchmark_tps.py` 的全部公开 CLI 参数。表中的默认值来自脚本；后续结果表使用的是各测试命令显式传入的值。

| 参数 | 默认值 | 适用后端 | 说明 |
|---|---|---|---|
| `--backend` | 必填 | 全部 | 选择 `minivllm`、`vllm` 或 `transformers`，每次只运行一个后端。 |
| `--max-input-tokens` | `128` | 全部 | 固定模式下的输入长度；随机模式下的最大输入长度。 |
| `--num-sequences` | `3` | 全部 | 每次 generation 包含的请求数量。 |
| `--max-output-tokens` | `256` | 全部 | 固定模式下的 `max_tokens`；随机模式下的最大输出限制。 |
| `--random-length` | 关闭 | mini-vLLM、vLLM | 开启后，输入长度和 `max_tokens` 分别在 `[ceil(max/8), max]` 内独立随机采样；Transformers 会忽略该参数。 |
| `--warmup-steps` | `2` | 全部 | 正式计时前执行的 generation 次数，不计入结果。 |
| `--repeat` | `1` | 全部 | 正式计时的 generation 次数，结果按全部 repeat 聚合。 |
| `--seed` | `0` | 全部 | workload 构造和每次计时 generation 使用的基础随机种子。 |
| `--respect-eos` | 关闭 | 全部 | 开启后允许遇到 EOS 提前结束；默认忽略 EOS。 |
| `--gpu-memory-utilization` | `0.9` | mini-vLLM、vLLM | 传给推理引擎的 GPU 显存利用率，取值范围为 `(0, 1]`。 |
| `--enforce-eager` | 关闭 | mini-vLLM、vLLM | 开启后禁用 CUDA Graph，使用 eager Decode。 |
| `--model-dtype` | `bfloat16` | 全部 | 模型 dtype，可选 `float32`、`float16` 或 `bfloat16`。 |

`--max-input-tokens`、`--num-sequences` 和 `--max-output-tokens` 必须为正数，并且最大输入与最大输出长度之和不能超过 Qwen3-0.6B 配置中的 `40960`。

TPS 统计口径：

```text
output TPS = 所有计时 repeat 的实际生成 token 总数 / 总生成耗时
```

输入 token 会增加耗时，但不计入 TPS 分子。

### 固定长度

每个 repeat 包含 64 条请求，每条请求固定输入 512 tokens，并固定生成 512 tokens。

#### mini-vLLM

```bash
python3 benchmarks/benchmark_tps.py \
  --backend minivllm \
  --max-input-tokens 512 \
  --num-sequences 64 \
  --max-output-tokens 512 \
  --warmup-steps 1 \
  --repeat 3 \
  --seed 0 \
  --model-dtype bfloat16 \
  --gpu-memory-utilization 0.9
```

#### vLLM V0

```bash
VLLM_USE_V1=0 python3 benchmarks/benchmark_tps.py \
  --backend vllm \
  --max-input-tokens 512 \
  --num-sequences 64 \
  --max-output-tokens 512 \
  --warmup-steps 1 \
  --repeat 3 \
  --seed 0 \
  --model-dtype bfloat16 \
  --gpu-memory-utilization 0.9
```

#### vLLM V1

```bash
VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 benchmarks/benchmark_tps.py \
  --backend vllm \
  --max-input-tokens 512 \
  --num-sequences 64 \
  --max-output-tokens 512 \
  --warmup-steps 1 \
  --repeat 3 \
  --seed 0 \
  --model-dtype bfloat16 \
  --gpu-memory-utilization 0.9
```

#### Transformers

```bash
python3 benchmarks/benchmark_tps.py \
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
| mini-vLLM | Decode CUDA Graph Replay | 18.7222 s | 98,304 | 6.2407 s | 32,768.00 | 5250.6584 tokens/s |
| vLLM | V0 | 20.3883 s | 98,304 | 6.7961 s | 32,768.00 | 4821.5941 tokens/s |
| vLLM | V1 | 15.5153 s | 98,304 | 5.1718 s | 32,768.00 | 6335.9521 tokens/s |
| Transformers | `generate()` | 101.1002 s | 98,304 | 33.7001 s | 32,768.00 | 972.3426 tokens/s |

在该固定长度 workload 下，vLLM V1 吞吐最高。mini-vLLM 达到 vLLM V1 的 **82.9%**，是 vLLM V0 的 **1.09x**、Transformers 的 **5.40x**，相对 vLLM V0 高 **8.9%**。

该结果说明 mini-vLLM 在这一规则的大 batch workload 下接近 vLLM V0，但仍落后于 vLLM V1。由于没有提供 `--enforce-eager` 对照，当前结果不能单独量化 CUDA Graph 的贡献，也不能将差距归因于某一项具体机制。

### 随机长度

启用 `--random-length` 后，每条请求的输入长度和 `max_tokens` 分别在 `[128, 1024]` 中独立均匀采样。Transformers Backend 当前忽略该选项，因此随机长度结果只比较 mini-vLLM、vLLM V0 和 vLLM V1。

#### mini-vLLM

```bash
python3 benchmarks/benchmark_tps.py \
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
```

#### vLLM V0

```bash
VLLM_USE_V1=0 python3 benchmarks/benchmark_tps.py \
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

#### vLLM V1

```bash
VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 benchmarks/benchmark_tps.py \
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
| mini-vLLM | Decode CUDA Graph Replay | 33.9433 s | 112,219 | 11.3144 s | 37,406.33 | 3306.0748 tokens/s |
| vLLM | V0 | 32.1858 s | 112,219 | 10.7286 s | 37,406.33 | 3486.6007 tokens/s |
| vLLM | V1 | 26.4411 s | 112,219 | 8.8137 s | 37,406.33 | 4244.1133 tokens/s |

三种引擎生成的 token 总数完全一致，说明逐请求 `max_tokens` workload 已正确对齐。mini-vLLM 达到 vLLM V1 的 **77.9%**、vLLM V0 的 **94.8%**；vLLM V0 比 mini-vLLM 高 **5.5%**。

随机输入和输出长度会使活跃 batch size 持续变化，可能影响 CUDA Graph batch 对齐、调度和 KV Cache 管理，但当前测试不能分离这些因素各自的影响。

固定长度与随机长度测试使用了不同的最大输入/输出长度，不能直接用两张表判断随机化本身带来的性能变化。两组测试均只进行了 1 次 warmup 和 3 次计时，没有报告方差。若要形成更稳定的结论，应补充相同最大长度下的固定/随机对照，增加 warmup、repeat 和独立重复运行，并报告离散程度。

## INT8 KV Cache

`benchmarks/benchmark_int8.py` 将三种模式放在独立子进程中依次运行，避免模型、NCCL、CUDA Graph 和 KV Cache 预分配相互影响。

| 模式 | 模型与激活 dtype | KV Cache dtype |
|---|---|---|
| FP16 | FP16 | FP16 |
| BF16 | BF16 | BF16 |
| INT8 | BF16 | INT8 数据 + FP32 scale |

INT8 与 BF16 的对比保持模型和激活 dtype 一致，用于隔离 KV Cache 量化的影响。吞吐计时使用随机非特殊 token ID、忽略 EOS，并统计完整 Prefill、Decode、调度、采样和文本解码；质量测试使用相同的 8 个自然语言 prompt、低温度采样和相同随机种子。

### INT8 Benchmark 参数

下表记录 `benchmarks/benchmark_int8.py` 的全部公开 CLI 参数。`_worker_*` 参数仅用于脚本内部启动子进程，不属于用户接口，因此不在表中列出。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-name-or-path` | `Qwen/Qwen3-0.6B` | Hugging Face 模型名或本地目录；当前硬编码模型配置仅适用于 Qwen3-0.6B。 |
| `--max-input-tokens` | `128` | 固定模式下的输入长度；随机模式下的最大输入长度。 |
| `--num-sequences` | `32` | 每次吞吐测试包含的请求数量。 |
| `--max-output-tokens` | `256` | 固定模式下的 `max_tokens`；随机模式下的最大输出限制。 |
| `--warmup-steps` | `1` | 每种模式正式计时前执行的 generation 次数。 |
| `--repeat` | `3` | 每种模式正式计时的 generation 次数。 |
| `--seed` | `0` | workload、吞吐采样和质量测试使用的基础随机种子。 |
| `--temperature` | `0.6` | 吞吐测试的采样温度，必须大于 `1e-10`。 |
| `--gpu-memory-utilization` | `0.9` | GPU 显存利用率，取值范围为 `(0, 1]`。 |
| `--cache-block-size` | `256` | KV Cache block size，必须是正的 2 的幂次。 |
| `--max-num-batched-tokens` | `16384` | Scheduler 每轮允许处理的最大 token 数。 |
| `--random-length` | 关闭 | 开启后，吞吐请求的输入长度和 `max_tokens` 分别在 `[ceil(max/8), max]` 内独立随机采样。 |
| `--respect-eos` | 关闭 | 开启后允许吞吐请求遇到 EOS 提前结束；默认忽略 EOS。 |
| `--enforce-eager` | 关闭 | 开启后禁用 mini-vLLM Decode CUDA Graph Replay。 |
| `--int8-model-dtype` | `bfloat16` | INT8 KV Cache 模式的模型与激活 dtype，可选 `float16` 或 `bfloat16`；相同 dtype 的非量化模式作为主要对照组。 |
| `--quality-max-tokens` | `64` | 每条质量测试 prompt 的最大生成 token 数。 |
| `--quality-temperature` | `0.1` | 质量测试的低温采样温度，必须大于 `1e-10`。 |
| `--quality-prompts-file` | 未指定 | 可选 JSON 文件；未指定时使用脚本内置的 8 条问答。文件内容可为 prompt 字符串列表，或包含 `prompt` 与 `accepted_answers` 的对象列表。 |
| `--show-samples` | `2` | 最终打印 completion 的质量测试样本数量；设为 `0` 可关闭样例输出。 |

长度、序列数、repeat、质量输出长度和 `max_num_batched_tokens` 必须为正数；`warmup_steps` 与 `show_samples` 可以为 `0`。最大输入与最大输出长度之和不能超过 `40960`。

### 固定长度

每个 repeat 包含 64 条请求，每条请求固定输入 512 tokens，并固定生成 512 tokens。

```bash
python3 benchmarks/benchmark_int8.py \
  --max-input-tokens 512 \
  --num-sequences 64 \
  --max-output-tokens 512 \
  --warmup-steps 1 \
  --repeat 3 \
  --seed 0 \
  --gpu-memory-utilization 0.9
```

| 模式 | 模型 dtype | KV Cache dtype | 总输出 token 数 | 单次平均延迟 | TPS | 相对 BF16 |
|---|---|---|---:|---:|---:|---:|
| FP16 | FP16 | FP16 | 98,304 | 5.773 s | 5675.65 tokens/s | +8.2% |
| BF16 | BF16 | BF16 | 98,304 | 6.248 s | 5244.32 tokens/s | 基线 |
| INT8 | BF16 | INT8 | 98,304 | 5.747 s | 5701.56 tokens/s | +8.7% |

固定长度 workload 下，INT8 比同为 BF16 模型计算的 BF16 KV Cache 快 **8.7%**，并比 FP16 模式快约 **0.5%**。当前端到端测试没有单独测量各 Kernel，不能进一步拆分 KV 带宽、量化和反量化的具体贡献。

### 随机长度

每条请求的输入长度与 `max_tokens` 分别在 `[128, 1024]` 中独立均匀采样。5 次计时中的实际输入长度为 `128/576.6/1024`，`max_tokens` 为 `128/584.7/1020`（最小值/平均值/最大值）。

```bash
python3 benchmarks/benchmark_int8.py \
  --max-input-tokens 1024 \
  --num-sequences 64 \
  --max-output-tokens 1024 \
  --warmup-steps 1 \
  --repeat 5 \
  --seed 0 \
  --gpu-memory-utilization 0.9 \
  --random-length
```

| 模式 | 模型 dtype | KV Cache dtype | 总输出 token 数 | 单次平均延迟 | TPS | 相对 BF16 |
|---|---|---|---:|---:|---:|---:|
| FP16 | FP16 | FP16 | 187,111 | 10.226 s | 3659.41 tokens/s | +7.3% |
| BF16 | BF16 | BF16 | 187,111 | 10.971 s | 3410.99 tokens/s | 基线 |
| INT8 | BF16 | INT8 | 187,111 | 10.880 s | 3439.66 tokens/s | +0.8% |

随机长度 workload 下，INT8 相对 BF16 的吞吐提升缩小到 **0.8%**，并比 FP16 模式低约 **6.0%**。活跃 batch size、上下文长度和请求结束时间持续变化，量化收益更容易被反量化及调度开销抵消，因此当前结果不支持“INT8 在所有 workload 下都能显著加速”的结论。

### 显存与缓存容量

mini-vLLM 会按照 `gpu_memory_utilization=0.9` 尽可能预分配 KV block，因此三种模式的 KV pool 和峰值 allocated memory 接近；INT8 的收益主要体现为相同显存预算下可以容纳更多 token。

| 模式 | 峰值 allocated memory | KV bytes/token | KV pool | 最大 block 数 | 最大缓存 token 数 | 固定 workload KV 上界 | 随机 workload KV 上界 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 35,929.6 MiB | 114,688 | 33,768.0 MiB | 1,206 | 308,736 | 7,168.0 MiB | 9,464.0 MiB |
| BF16 | 35,929.6 MiB | 114,688 | 33,768.0 MiB | 1,206 | 308,736 | 7,168.0 MiB | 9,464.0 MiB |
| INT8 | 35,945.6 MiB | 59,136 | 33,783.8 MiB | 2,340 | 599,040 | 3,696.0 MiB | 4,879.9 MiB |

INT8 将 KV 存储从 `114,688` 降至 `59,136 bytes/token`，降低 **48.4%**；最大缓存容量从 `308,736` 提高到 `599,040 tokens`，增加 **94.0%**。峰值 allocated memory 增加约 16 MiB，是相同显存利用率下预分配 block 数增多的结果，不代表 INT8 没有降低单 token KV 空间。

### 生成质量

固定和随机长度命令使用相同的独立质量测试，因此结果只列一次。BF16 是同模型 dtype 参考。

| 模式 | 简短问答准确率 | 与 BF16 完全一致 | token 位置一致率 | 编辑相似度 | 最长公共前缀比例 | 输出长度比例 |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 75.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| BF16 | 75.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| INT8 | 75.0% | 87.5% | 96.6% | 99.0% | 98.1% | 98.3% |

INT8 在 8 个简短问答上没有降低任务命中率，并与 BF16 保持 `96.6%` 的 token 位置一致率和 `99.0%` 的编辑相似度。该测试只是生成稳定性和基础问答的冒烟测试，样本量很小，不能替代标准数据集上的困惑度或下游任务评测。

综合来看，当前 INT8 KV Cache 的主要稳定收益是接近减半的 KV 存储和接近翻倍的缓存容量；吞吐在固定长度场景提升明显，在随机长度场景基本持平；小型质量测试未观察到任务准确率下降。

## 结果限制

- 所有结果只适用于本文记录的硬件、软件版本、模型和 workload。
- Attention 微基准的 PyTorch 路径是朴素数值参考，不能代表生产级 PyTorch SDPA 或 FlashAttention 性能。
- 端到端测试的 warmup 和 repeat 数量有限，未报告方差或跨进程重复结果。
- mini-vLLM、vLLM 和 Transformers 的文本解码口径尚未完全统一。
- INT8 质量测试只有 8 个样本，不足以形成通用的生成质量结论。
