# Mini-vLLM

轻量级大语言模型推理引擎，实现现代 LLM Serving 系统中的核心优化路径。

项目围绕 LLM 推理优化展开，实现请求调度、Paged KV Cache、Triton Attention Kernel、Decode CUDA Graph Replay、Tensor Parallel 和 INT8 KV Cache 量化等核心模块，并在 Qwen3-0.6B 上完成端到端性能验证。

[核心技术](#核心技术与实现) · [快速开始](#快速开始) · [性能测试](#性能测试) · [Benchmark 详细记录](docs/benchmark.md) · [实现原理与代码导读](docs/learn.md)

## 核心技术与实现

| 模块 | 关键实现                                                        | 优化目标 |
|---|-------------------------------------------------------------|---|
| 推理调度 | Continuous Batching、Chunked Prefill、Request Preemption      | 动态组织 Prefill/Decode，提高 GPU 利用率 |
| KV Cache | Paged KV Cache、Prefix Cache、block 生命周期管理                    | 降低显存碎片，提高缓存复用率 |
| Attention | Triton FlashAttention、PagedAttention、Online Softmax、GQA/MQA | 降低 Attention 计算和分页 K/V 访问开销 |
| KV Cache 量化 | Token-wise + KV-head-wise 动态对称 INT8 Quantization            | 降低 KV 存储，提高缓存容量 |
| 推理执行 | Decode CUDA Graph Replay、多 batch size Graph 复用                    | 降低 Decode Kernel Launch 开销 |
| 分布式推理 | Tensor Parallel、NCCL 通信（未多卡实测）                              | 提供模型切分和多 GPU 扩展路径 |
| 模型加载 | 模型工厂、Transformers Tokenizer、safetensors 权重加载                | 统一 Qwen3/Llama 模型构建与权重加载路径 |

## 系统架构

```text
           用户请求
              |
              v
          Tokenizer
              |
              v
          LLM Engine
              |
              v
          Scheduler
              |
   +----------+----------+
   |                     |
   v                     v
Sequence 队列        BlockManager
(waiting / running)      |
   |                     |
   +----------+----------+
              |
              v
         ModelRunner
              |
              v
       Prefill / Decode
              |
              v
        Model Forward
              |
              v
   Triton Attention Kernel
              |
              v
        Paged KV Cache
              |
              v
Unquantized KV Cache / INT8 Quantized KV Cache
              |
              v
        Model Output
              |
              v
      LM Head + Sampler
              |
              v
         Output Token
```

`Scheduler` 负责请求队列、Continuous Batching、Chunked Prefill 和抢占；
`BlockManager` 管理物理 block 编号、引用计数、Prefix Cache 和 `Sequence.block_table`，但不直接保存 K/V Tensor;
`ModelRunner` 负责 batch 输入构造、模型 forward、采样流程以及 Decode 阶段 CUDA Graph 捕获与复用。

## 快速开始

项目依赖 Linux、支持 CUDA 的 NVIDIA GPU，以及与 CUDA 环境匹配的 PyTorch。先安装 PyTorch，再安装其余运行依赖：

```bash
git clone https://github.com/lvyy1999/mini-vllm
cd mini-vllm

python3 -m pip install transformers triton tqdm numpy xxhash packaging
```

运行 Qwen3-0.6B 推理示例：

```bash
python3 main.py
```

`main.py` 会加载 Qwen3-0.6B，批量处理三个聊天 prompt，并打印 prompt 与 completion。首次运行需要从 Hugging Face 下载 tokenizer、配置和 safetensors 权重；离线运行时，可将脚本中的 `model` 改为完整的本地模型目录，并保留目录名 `Qwen3-0.6B` 供模型工厂识别。

运行 Llama-3.2-1B-Instruct 推理示例：

```bash
python3 main_llama32.py
```

`main_llama32.py` 会从本地目录加载 Llama-3.2-1B-Instruct，批量处理三个聊天 prompt，并打印 prompt 与 completion。首次运行需要从 Hugging Face 下载 tokenizer、配置和 safetensors 权重。

### 启用 INT8 KV Cache

KV Cache 默认使用 `auto` 并跟随模型 dtype。构造 `LLM` 时传入 `kv_cache_dtype="int8"` 即可启用 INT8：

```python
llm = LLM(
    enforce_eager=True,
    model_name_or_path=model,
    custom_model_config=model_config,
    kv_cache_dtype="int8",
)
```

当前量化实现具有以下特征：

- 对每个 token、每个 KV head 独立计算对称量化 scale。
- K Cache 和 V Cache 使用 INT8 数据，并分别保存 FP32 scale。
- Cache 写入时动态量化，Prefill 和 Decode Attention 计算时动态反量化。
- 仅改变 KV Cache 存储格式，不量化模型权重或激活。

## 性能测试

### 核心测试结果

| 测试 | 核心结果 |
|---|---|
| Prefill FlashAttention | 相对朴素 GPU PyTorch 参考实现加速 **7.61x-28.26x** |
| Decode PagedAttention | 相对朴素 GPU PyTorch 参考实现加速 **5.57x-54.12x** |
| 端到端输出吞吐 | mini-vLLM 达到 `5250.66 tokens/s`，是 vLLM V0 的 **1.09x**、Transformers 的 **5.40x**，达到 vLLM V1 的 **82.9%** |
| INT8 KV Cache 吞吐 | 相对 BF16 KV Cache，固定长度提升 **8.7%**，随机长度提升 **0.8%** |
| INT8 KV Cache 存储 | 单 token KV 存储降低 **48.4%**，相同显存预算下最大缓存容量提高 **94.0%** |
| INT8 质量冒烟测试 | 8 条问答准确率与 BF16 持平，token 位置一致率为 **96.6%** |

测试环境：

| 项目 | 配置 |
|---|---|
| 平台 | AutoDL，单 GPU |
| GPU | NVIDIA A100 PCIe 40GB |
| 模型 | Qwen/Qwen3-0.6B，BF16 |
| 软件 | PyTorch 2.6.0+cu124、Triton 3.1.0 |

### 端到端吞吐

| 推理后端 | 模式 | TPS |
|---|---|---:|
| mini-vLLM | Decode CUDA Graph Replay | 5250.66 tokens/s |
| vLLM | V0 | 4821.59 tokens/s |
| vLLM | V1 | 6335.95 tokens/s |
| Transformers | `generate()` | 972.34 tokens/s |

本次测试中，mini-vLLM 吞吐高于 vLLM V0，低于 vLLM V1。

### INT8 KV Cache

仅改变 KV Cache 存储格式，模型权重和计算激活保持原始精度。

| 指标 | BF16 KV Cache | INT8 KV Cache | 变化 |
|---|---:|---:|---:|
| 固定长度 TPS | 5244.32 | 5701.56 | +8.7% |
| 随机长度 TPS | 3410.99 | 3439.66 | +0.8% |
| KV 存储 | 114,688 bytes/token | 59,136 bytes/token | -48.4% |
| 最大缓存容量 | 308,736 tokens | 599,040 tokens | +94.0% |
| 8 条简短问答准确率 | 75.0% | 75.0% | 持平 |

> 结果基于特定 workload，用于验证当前实现的优化效果；详细测试口径和限制见 [Benchmark 详细记录](docs/benchmark.md)。

## 项目结构

```text
mini-vllm/
├── src/minivllm/
│   ├── models/                 # 模型工厂与 Qwen3、Llama 3.2 实现
│   ├── engine/                 # 调度器、KV Cache、执行引擎与 ModelRunner
│   ├── layers/                 # 模型基础层与自定义算子
│   │   ├── attention.py        # Triton Attention Kernel 与 INT8 KV Cache 量化
│   │   └── ...                 # Linear、RoPE、RMSNorm、Sampler 等组件
│   ├── utils/                  # 配置、运行上下文与权重加载
│   ├── llm.py                  # 顶层 LLM 接口
│   └── sampling_parameters.py  # 采样参数
├── benchmarks/
│   ├── benchmark_prefilling.py # Prefill Attention 微基准
│   ├── benchmark_decoding.py   # Decode Attention 微基准
│   ├── benchmark_tps.py        # 端到端输出 TPS Benchmark
│   └── benchmark_int8.py       # KV Cache 吞吐、显存与质量 Benchmark
├── docs/
│   ├── learn.md                # 实现原理与代码导读
│   └── benchmark.md            # 完整 Benchmark 记录
├── tests/                      # 单元测试与中文测试文档
├── main.py                     # Qwen3-0.6B 推理示例
└── main_llama32.py             # Llama-3.2-1B-Instruct 推理示例
```

## 单元测试

安装 pytest 并在项目根目录运行：

```bash
python3 -m pip install pytest
python3 -m pytest tests -v
```

当前测试套件共 159 项，已在 CUDA 环境中全部通过。测试覆盖配置、采样、Sequence、Block Manager、Scheduler、核心模型层、权重加载、模型工厂、Model Runner、LLM Engine，以及 INT8 KV Cache 的写入、Decode 和 Chunked Prefill。

每个 `test_*.py` 都有同名的 `test_*.md` 中文文档，记录单独运行命令、覆盖范围和 CUDA 依赖。

## 文档

- [实现原理与代码导读](docs/learn.md)：按算子、模型、KV Cache、调度、执行引擎和 Benchmark 组织的学习路线。
- [Benchmark 详细记录](docs/benchmark.md)：完整环境、命令、结果表、测试口径与限制。
- [单元测试文档](tests)：各测试模块的中文覆盖说明。

## 参考项目

- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [Triton](https://github.com/triton-lang/triton)
