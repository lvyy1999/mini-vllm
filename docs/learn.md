<h1 align="center">mini-vLLM 学习路线</h1>

本文档基于仓库当前代码，按“算子 -> 模型 -> KV Cache -> 调度 -> 执行引擎 -> Benchmark”的顺序介绍一个轻量级 LLM 推理引擎。项目面向推理而不是训练，当前以 Qwen3-0.6B、Llama-3.2-1B-Instruct 为实测模型，引擎实现了 Paged KV Cache、前缀缓存、chunked prefill、可选 INT8 KV Cache、Triton Attention、Decode CUDA Graph Replay，以及张量并行代码路径。

> 核心代码位于 `src/minivllm`。当前 `ModelRunner` 无论单卡还是多卡都会初始化 NCCL，因此运行环境需要 CUDA、NCCL，以及支持 CUDA 的 PyTorch。张量并行/NCCL 代码路径尚未在真实多 GPU 环境中完整验证，相关章节描述的是当前代码设计。

## 0. 总体架构

一次生成请求的主调用链如下：

```text
LLM.generate()  # 继承自 LLMEngine
  -> add_prompt()
     -> Sequence
     -> Scheduler.waiting
  -> step()
     -> Scheduler.schedule()
     -> ModelRunner.call("run", ...)
        -> prepare_prefill() / prepare_decode()
        -> model.forward()
        -> compute_logits()
        -> SamplerLayer
     -> Scheduler.postprocess()
```

核心模块分工：

| 模块 | 作用 |
|---|---|
| `layers/` | RMSNorm、张量并行 Linear、RoPE、Triton Attention、INT8 KV Cache 量化/反量化、采样 |
| `models/` | Qwen3、Llama 3.2 模型结构，以及统一的模型工厂 |
| `engine/sequence.py` | 保存单条请求的 token、状态和调度进度 |
| `engine/block_manager.py` | 分配、复用和释放分页 KV Cache block |
| `engine/scheduler.py` | 在 prefill、chunked prefill 和 decode 之间调度 |
| `engine/model_runner.py` | 准备输入、运行模型、分配不同 dtype 的 KV Cache、管理 CUDA Graph 和多卡通信 |
| `engine/llm_engine.py` | 对外生成接口和整个推理循环 |

## Step 1：基础层

### 1.1 SiLUAndMul

实现：[activation.py](../src/minivllm/layers/activation.py)

当前激活层实现的是 SwiGLU 中常见的融合形式，而不是一组 SiLU/GELU 激活函数：

```python
gate, up = x.chunk(2, dim=-1)
return F.silu(gate) * up
```

Qwen3 和 Llama 的 `gate_proj`、`up_proj` 被合并为一次列并行线性计算，输出再交给 `SiLUAndMul`。该函数使用 `torch.compile`，目的是融合逐元素操作并减少 kernel 启动和中间张量开销。

`torch.compile` 的收益与输入规模、shape 稳定性、PyTorch/Triton 版本和 GPU 有关。首次编译成本通常应通过 warmup 排除，不能简单归因于“小张量编译更慢”。

### 1.2 RMSNorm 与残差融合

实现：[rmsnorm.py](../src/minivllm/layers/rmsnorm.py)

RMSNorm 不减均值，只根据最后一维的均方根缩放：

```text
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```

当前类提供两条路径：

- `rms_forward(x)`：普通 RMSNorm。
- `residual_rms_forward(x, residual)`：先执行 `x + residual`，再归一化，同时返回归一化前的和作为下一层 residual。

```python
origin_dtype = x.dtype
x = x.float().add_(residual.float())
residual = x.to(origin_dtype)
var = x.pow(2).mean(dim=-1, keepdim=True)
x.mul_(torch.rsqrt(var + self.eps))
x = x.to(origin_dtype).mul_(self.weight)
return x, residual
```

实际实现使用 FP32 完成残差相加和归一化统计，再转换回输入 dtype。这种接口让 DecoderLayer 可以沿层传递 residual，减少重复的残差处理代码。`weight` 是 `nn.Parameter`，可从 Hugging Face checkpoint 加载。

### 1.3 张量并行 Linear

实现：[linear.py](../src/minivllm/layers/linear.py)

这些层用于张量并行推理：

| 类型 | 切分方式 | 前向通信 |
|---|---|---|
| `ReplicatedLinear` | 不切分 | 无 |
| `ColumnParallelLinear` | 按输出维度切分 | 无 |
| `RowParallelLinear` | 按输入维度切分 | `dist.all_reduce` |
| `MergedColumnParallelLinear` | 合并多个列并行矩阵 | 无 |
| `QKVColumnParallelLinear` | 分别切分 Q/K/V heads | 无 |

权重加载器接收完整 checkpoint Tensor，然后只复制当前 TP rank 对应的 shard。对于合并权重，模型的 `packed_module_mapping` 将 checkpoint 中独立的 `q_proj/k_proj/v_proj` 和 `gate_proj/up_proj` 映射到合并参数中的不同区域。

`RowParallelLinear` 只在 rank 0 加 bias，随后对各 rank 的局部结果求和，避免 bias 被重复累加。

### 1.4 VocabParallelEmbedding 与 LM Head

实现：[embedding_head.py](../src/minivllm/layers/embedding_head.py)

`VocabParallelEmbedding` 按词表维度切分权重：

1. 每个 rank 只保存一段词表。
2. 不属于本 rank 的 token 先映射到安全索引，再将输出 mask 为 0。
3. 多卡通过 `dist.all_reduce` 得到完整 embedding。

`ParallelLMHead` 在每个 rank 上计算局部词表 logits，然后使用 `dist.gather` 只把完整 logits 收集到 rank 0。接收缓冲区通过 `torch.empty_like(logits)` 创建，因此 shape、dtype 和 device 都与局部 logits 一致。

Prefill 时不需要为每个输入 token 计算词表 logits。LM Head 根据 `cu_seqlens_q` 只选择每条序列本轮最后一个 query 的 hidden state。

### 1.5 Triton Attention

实现：[attention.py](../src/minivllm/layers/attention.py)

当前不是调用第三方 FlashAttention，而是实现了四个关键路径：

| 路径 | 输入 | 用途 |
|---|---|---|
| `store_kvcache` | 当前 K/V、`slot_mapping` | 把新 K/V 直接写入缓存，或量化后写入 INT8 缓存 |
| prefill without cache | 连续 Q/K/V | 普通首次 prefill |
| prefill with cache | Q、Paged K/V、`block_tables` | chunked prefill 或前缀复用 |
| decode with cache | 单 token Q、Paged K/V | 自回归解码 |

#### Paged KV Cache

单层缓存布局为：

```text
(num_blocks, block_size, num_kv_heads, head_dim)
```

逻辑 token 位置先通过 `block_tables` 找到物理 block，再通过 block 内偏移找到实际 K/V：

```text
logical_block = token_position // block_size
block_offset  = token_position % block_size
physical_block = block_tables[sequence, logical_block]
```

`slot_mapping` 负责写入，`block_tables` 负责读取，两者不要混淆。

with-cache kernel 会把尾部无效 lane 的逻辑 block 索引限制在合法范围，再读取 block table；真正读取 K/V 时仍由 `kv_mask` 屏蔽无效位置。这样既不会访问越界地址，也避免了 Triton 3.1 中 masked block-table load 可能触发的布局冲突。

#### INT8 KV Cache

配置项 `kv_cache_dtype` 默认为 `auto`，此时 KV Cache 跟随模型 dtype。设置为 `int8` 后，缓存数据改为 INT8，并为 K、V 分别保存 FP32 scale。量化粒度是“每个 token、每个 KV head、每个 K/V 向量一组 scale”，属于动态对称量化，不使用 zero point：

```text
scale = max(abs(x)) / 127
q = clip(round(x / scale), -127, 127)
x_dequant = q * scale
```

全零向量的 scale 设为 `1.0`，避免除零。当前 Triton 写入 kernel 使用四舍五入到最近整数，恰好位于中点时远离 0，然后转为 `tl.int8`。

INT8 数据缓存的单层形状不变：

```text
K/V data:  (num_blocks, block_size, num_kv_heads, head_dim)
K/V scale: (num_blocks, block_size, num_kv_heads)
```

执行路径如下：

- `store_kvcache` 在 FP32 中计算绝对最大值与 scale，再将 K/V 量化写入 INT8 cache。
- 普通首次 prefill 仍直接使用当前模型 dtype 的 K/V 计算 Attention，同时把量化副本写入缓存。
- chunked prefill 和 decode 从缓存加载 INT8 K/V，在 FP32 中乘以 scale，随后转换为 query dtype 参与 Attention。
- 模型权重、激活和 Attention 输出仍使用模型 dtype；INT8 只作用于 KV Cache。

#### GQA/MQA head 映射

当 `num_heads > num_kv_heads` 时，多个 query head 共享一个 KV head：

```python
kv_head_idx = head_idx // (num_heads // num_kv_heads)
```

#### Online Softmax

Prefill kernel 不保存完整的 `N x N` attention matrix，而是按 K/V tile 更新行最大值 `m_i`、指数和 `l_i` 与输出累加器 `acc`。这样避免了完整 attention score 的显存写入。

`tl.dot` 的两个输入必须 dtype 一致。当前代码在计算 `P @ V` 前使用：

```python
tl.dot(p_ij.to(v_j.dtype), v_j)
```

Decode kernel 也已经采用分块读取：每个 program 负责一个 `(batch, query head)`，按 `BLOCK_N` 生成分页地址，一次加载形状为 `(BLOCK_N, head_dim)` 的 K/V，并使用 online softmax 累加输出。因此它不是逐 token、逐行加载 K/V 的旧实现；后续优化重点应放在 tile、warp 数、寄存器压力和分页访存效率上。

#### Tile 选择

Prefill wrapper 会读取当前 GPU 的 compute capability，并结合 `head_dim` 选择 tile。当前逻辑是针对无 dropout、causal Attention 的简化启发式选择：

| GPU 条件 | `head_dim` | `BLOCK_M` | `BLOCK_N` |
|---|---:|---:|---:|
| compute capability < 8.0 | `<= 64` | 64 | 64 |
| compute capability < 8.0 | `65-128` | 32 | 32 |
| compute capability < 8.0 | `> 128` | 16 | 16 |
| A100（SM80） | `<= 64` | 128 | 128 |
| A100（SM80） | `> 64` | 128 | 64 |
| 其他 compute capability >= 8.0 | `<= 64` | 128 | 128 |
| 其他 compute capability >= 8.0 | `> 64` | 64 | 64 |

Decode 每个 program 只处理一个 query，因此没有 `BLOCK_M`；`head_dim <= 64` 时 `BLOCK_N=128`，否则 `BLOCK_N=64`。这些值不是跨设备的理论最优解，调整后仍应在目标 GPU、序列长度和 dtype 上重新 benchmark。

#### Triton program 与线程

Triton 的 `program_id` 表示一个 program instance，而不是 CUDA 单线程。Prefill grid 为：

```text
(num_heads, ceil(max_seqlen_q / BLOCK_M), num_sequences)
```

Decode grid 为：

```text
(num_heads, batch_size)
```

实际每个 program 使用多少 warp 由 Triton 编译配置决定，不能把“一个 grid/program”固定解释成 4 个 warp 或 128 个线程。

### 1.6 RoPE

实现：[rotary_embedding.py](../src/minivllm/layers/rotary_embedding.py)

RoPE 将绝对位置转换成 Q/K 的旋转角度，使注意力分数包含相对位置信息。缓存 `cos_sin_cache` 在模型初始化时一次创建，前向时按 `positions` 索引。

当前实现支持：

- Qwen3：标准 RoPE，使用模型配置中的 `rope_theta`。
- Llama 3.2：`is_llama3=True`，使用 `RotaryEmbedding` 中固定的默认 factor、high/low frequency factor 和原始上下文长度调整频率；当前模型工厂没有解析模型配置中的 `rope_scaling`。

当前代码没有实现通用 YaRN 或动态 NTK scaling，因此不应把它们列为已经完成的功能。

### 1.7 采样

实现：[sampler.py](../src/minivllm/layers/sampler.py)

rank 0 对 logits 除以每条序列的 temperature，执行 softmax 后使用指数噪声竞争采样：

```python
probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(-1)
```

`SamplingParams` 当前支持 `temperature`、`max_tokens` 和 `ignore_eos`，不允许 temperature 接近 0，因此没有 greedy sampling 模式。

## Step 2：模型构建

实现：[model_factory.py](../src/minivllm/models/model_factory.py)、[qwen3.py](../src/minivllm/models/qwen3.py)、[llama.py](../src/minivllm/models/llama.py)

当前模型工厂包含两个模型入口，分别为 Qwen3-0.6B 和 Llama-3.2-1B-Instruct：

| 模型 | 入口 | 主要差异 |
|---|---|---|
| Qwen3-0.6B | `Qwen3ForCausalLM` | Q/K RMSNorm、Qwen RoPE 参数 |
| Llama-3.2-1B-Instruct | `LlamaForCausalLM` | 无 Q/K Norm、固定默认参数的 Llama 3.2 RoPE scaling、可配置 MLP bias |

两个模型都采用：

```text
VocabParallelEmbedding
  -> DecoderLayer x N
     -> RMSNorm + SelfAttention
     -> RMSNorm + MLP
  -> final RMSNorm
  -> ParallelLMHead
```

每个 TP rank 保存部分 query heads 和 KV heads，在本地完成 RoPE、GQA 和 Attention；输出投影通过 `RowParallelLinear` 的 `all_reduce` 恢复完整 hidden state。

`ModelRunner` 不再直接导入具体模型类，而是调用 `create_model(model_name_or_path, custom_model_config)`。模型工厂先用 `Path(model_name_or_path).name` 取得目录名，再从 `_MODEL_BUILDERS` 查找对应 builder，因此当前路径仍必须以 `Qwen3-0.6B` 或 `Llama-3.2-1B-Instruct` 结尾。

每个 builder 返回 `(model, RuntimeModelInfo)`。`RuntimeModelInfo` 统一提供 `num_layers`、`num_kv_heads`、`head_dim` 和 `hidden_size`，供 `ModelRunner` 估算 KV Cache、分配 CUDA Graph buffer。当前 `_MODEL_BUILDERS` 仍是显式字典，不是自动注册机制；新增模型时需要新增模型实现、builder 和字典项。

## Step 3：Sequence 与 KV Cache

### 3.1 Sequence

实现：[sequence.py](../src/minivllm/engine/sequence.py)

`Sequence` 保存的不只是 token 列表，还包含调度和缓存进度：

| 字段 | 含义 |
|---|---|
| `token_ids` | prompt 与已生成 token |
| `num_prompt_tokens` | 原始 prompt 长度 |
| `num_cached_tokens` | 已经写入 KV Cache 的 token 数 |
| `num_scheduled_tokens` | 本轮将处理的 token 数 |
| `is_prefill` | 当前是 prefill 还是 decode |
| `block_table` | 逻辑 block 到物理 block 的映射 |
| `temperature/max_tokens/ignore_eos` | 当前请求的采样参数 |

构造时使用 `copy(token_ids)`，避免外部修改原列表影响序列内部状态。

`__getstate__` 和 `__setstate__` 用于 pickle。rank 0 通过共享内存把 Sequence 发送给 worker：prefill 发送所需 token 列表，decode 只发送最后一个 token，从而减少进程间序列化数据量。

### 3.2 Block 与前缀缓存

实现：[block_manager.py](../src/minivllm/engine/block_manager.py)

每个 `Block` 记录：

- `block_id`：物理块编号。
- `ref_count`：共享该块的序列数量。
- `hash`：包含前缀信息的 xxHash64。
- `token_ids`：用于确认内容，防止只依赖 hash 时发生碰撞误命中。

只有完整且已经完成计算的 block 才会记录前缀 hash。分配时，当前序列的最后一个逻辑 block 始终不作为缓存命中复用，无论它是否恰好填满，以保留后续写入空间；已填满并记录 hash 的 block 只有在更长请求中位于非末尾前缀位置时才可能被复用。

`can_allocate(seq)` 返回可复用的 cached token 数；显存不足时返回 `-1`。`allocate(seq)` 根据连续前缀命中情况增加引用计数、复用空闲旧 block，或分配新 block。

### 3.3 追加与释放

`can_append(seq)` 判断下一轮 decode 是否需要新 block。由于生成 token 会先追加到 Sequence、下一轮才计算其 KV，当：

```python
seq.num_tokens % block_size == 1
```

时说明新 token 已经跨入下一个逻辑 block，需要先分配物理 block。

`deallocate(seq)` 按逆序减少引用计数；引用计数变为 0 的 block 回到空闲队列，但 hash/token 信息会保留到该物理块真正被覆盖，因此仍有机会被前缀缓存复用。

## Step 4：Scheduler

实现：[scheduler.py](../src/minivllm/engine/scheduler.py)

Scheduler 维护 `waiting` 和 `running` 两个队列，并且每轮只返回一种阶段：prefill 或 decode。

### 4.1 Prefill 与 chunked prefill

Scheduler 优先处理 waiting 队列：

1. 调用 `BlockManager.can_allocate()` 检查显存并计算前缀命中长度。
2. 调用 `allocate()` 建立 block table。
3. 在 `max_num_batched_tokens` 预算内设置 `num_scheduled_tokens`。
4. prompt 未全部处理时保留在 waiting 队首，下一轮继续 chunked prefill。
5. prompt 全部处理完后将 Sequence 移入 running。

当前限制是：如果剩余 token 预算装不下完整 prompt，只允许本轮第一个序列执行 chunked prefill，后续 waiting 序列留到下一轮。

### 4.2 Decode 与抢占

只有本轮没有 prefill 请求时才调度 decode。每条 running Sequence 一次调度一个 token；如果新 block 不够，会从 running 队尾抢占序列，释放其 KV Cache，并把它放回 waiting 队首重新 prefill。

### 4.3 后处理

`postprocess()` 的顺序很重要：

1. 为本轮新填满的 block 计算 hash。
2. 把 `num_scheduled_tokens` 累加到 `num_cached_tokens`。
3. chunked prefill 尚未结束时，不追加采样 token。
4. prefill 完成或 decode 完成时，追加一个生成 token。
5. 命中 EOS 或达到 `max_tokens` 后释放 block 并移出 running。

## Step 5：ModelRunner

实现：[model_runner.py](../src/minivllm/engine/model_runner.py)

### 5.1 初始化、device 与 dtype

每个 rank 都会初始化 NCCL process group，并将当前 CUDA device 设置为该 rank。模型 dtype 从 `custom_model_config["torch_dtype"]` 读取，当前支持：

```text
float16 / bfloat16 / float32
```

模型创建、权重加载、warmup 和 CUDA Graph hidden-state 输出缓冲区使用 `self.model_dtype`。KV Cache dtype 由 `kv_cache_dtype` 独立控制：`auto` 跟随模型 dtype，`int8` 使用 INT8 数据缓存与 FP32 scale 缓存。INT8 模式不会改变模型权重或激活的 dtype。初始化完成后，全局默认 device/dtype 会恢复为 CPU 和进入 ModelRunner 前的默认 dtype。

启用 INT8 KV Cache 时，在构造 `LLM` 时传入：

```python
llm = LLM(..., kv_cache_dtype="int8")
```

在 Tesla T4 上应使用 `float16`，不要使用 `bfloat16`。

### 5.2 权重加载

实现：[loader.py](../src/minivllm/utils/loader.py)

加载器支持本地目录和 Hugging Face 模型名。远端下载仅保留 safetensors 和 JSON 文件，但实际权重加载只遍历 `.safetensors`；JSON 不由该加载函数解析。权重先在 CPU 侧读取，再由参数自身的 `weight_loader` 拷贝或切分到已经位于 GPU 的模型参数。

对于 QKV 和 gate/up 合并参数，`packed_module_mapping` 决定 checkpoint 权重应写入合并参数的哪一段。

### 5.3 Warmup 与 KV Cache 容量

`warmup_model()` 使用受 `max_num_batched_tokens`、`max_model_length` 和 `max_num_sequences` 限制的虚拟 batch 运行一次 prefill，记录峰值显存。

每个物理 KV block 的字节数按实际 cache dtype 计算：

```text
data_bytes = head_dim * kv_cache_dtype.itemsize
scale_bytes = 4 if kv_cache_dtype == int8 else 0
block_bytes = 2 * num_layers * block_size * num_kv_heads
              * (data_bytes + scale_bytes)
```

其中 `2` 表示 K 和 V，INT8 模式下额外的 `4` 字节来自每个 token、每个 KV head 的 FP32 scale。`RuntimeModelInfo.num_kv_heads` 保存模型的总 KV head 数，`ModelRunner` 初始化时除以 `world_size`，将 `self.num_kv_heads` 统一为每个 rank 的本地 KV head 数。block 字节估算和实际缓存分配都使用这个本地值。

每个 rank 上的 KV Cache 形状为：

```text
(2, num_layers, max_cache_blocks, block_size, num_kv_heads, head_dim)
```

INT8 模式还会分配 FP32 scale cache：

```text
(2, num_layers, max_cache_blocks, block_size, num_kv_heads)
```

各 rank 根据本地剩余显存计算 block 数，再使用 `all_reduce(MIN)` 取所有 rank 都能容纳的最小值，避免 Scheduler 分配超过某张卡容量的 block。随后每个 Attention 层持有属于自己的 K/V 数据与 scale 视图；Attention wrapper 从 `k_cache.shape[2]` 取得相同的本地 KV head 数，因此缓存 shape、分页步长和 kernel 寻址保持一致。`QKVColumnParallelLinear` 会校验总 Q/KV head 数能否被 TP world size 整除。

### 5.4 prepare_prefill

本轮每条序列只准备区间：

```text
[num_cached_tokens, num_cached_tokens + num_scheduled_tokens)
```

因此：

- `input_ids` 只包含本轮 query tokens。
- `positions` 使用序列中的绝对位置，保证分块后 RoPE 不会从 0 重新开始。
- `cu_seqlens_q` 表示本轮 query 边界。
- `cu_seqlens_k` 表示“历史缓存 + 本轮 query”的完整 KV 长度。
- `slot_mapping` 指定本轮 K/V 的写入位置。
- `block_tables` 在 chunked prefill 需要读取历史缓存时提供。

当总 query 长度小于总 KV 长度时，说明存在历史缓存，Attention 走 with-cache kernel。因果 mask 使用 query 在完整序列中的绝对起点：

```text
actual_start_q = full_seq_len - batch_len_q
```

### 5.5 prepare_decode

Decode 每条序列只输入 `last_token`，位置为 `len(seq) - 1`。同时准备：

- `context_lens`：包括当前 token 在内的 KV 长度。
- `slot_mapping`：当前 token K/V 的物理写入位置。
- `block_tables`：读取该序列全部分页 K/V 的映射。

### 5.6 共享内存与多进程

rank 0 负责 Scheduler 和采样，其余 rank 运行 `ModelRunner.loop()`：

1. rank 0 将方法名和参数 pickle 到名为 `minivllm` 的共享内存。
2. 每个 worker 对应一个 `Event`，rank 0 逐个 `set()`。
3. worker 读取 4 字节 payload 长度，再反序列化参数并调用相同方法。
4. 模型中的 NCCL collective 保证各 rank 在前向计算时同步。

这套多进程与 collective 路径已经写入代码，但尚未在真实多 GPU 环境中完成端到端验证。

### 5.7 CUDA Graph

Prefill shape 随 token 数变化，因此直接 eager 执行。Decode 每条序列固定输入一个 token，在 `enforce_eager=False` 且 batch 不超过 512 时使用 CUDA Graph。

基础捕获 batch size 为：

```python
[1, 2, 4, 8] + list(range(16, max_graph_bs + 1, 8))
```

如果 `max_graph_bs` 不在上述基础捕获列表中，代码会把它额外追加到 `self.graph_bs`，保证所有满足 `bs <= max_graph_bs` 的实际 batch 都能找到可用图。例如 `max_graph_bs=100` 时，列表末尾会包含 `96, 100`。

运行时从升序的 `self.graph_bs` 中选择第一个满足 `graph_bs >= bs` 的图，也就是能容纳实际 batch 的最小图。Replay 前会：

1. 把真实的 `input_ids` 和 `positions` 复制到静态 buffer 的前 `bs` 个位置。
2. 将所选图范围 `slot_mapping[:graph_bs]` 全部设置为 `-1`，再复制真实的 `slot_mapping[:bs]`。Triton KV 写入 kernel 遇到 `-1` 会直接返回，因此 padding token 不会覆盖 KV Cache。
3. 将 `context_lens[:graph_bs]` 清零，再复制真实序列长度，使 padding decode 不读取历史 KV。
4. Replay 后只取 `outputs[:bs]`，再执行 LM Head 和采样。

CUDA Graph 的 hidden-state 输出 buffer 显式使用 `self.model_dtype`，因此能够和 FP16/BF16 模型保持一致。`enforce_eager=True` 会完全绕过 Decode CUDA Graph Replay 路径。

捕获阶段的 `slot_mapping` 已使用 `torch.full(..., -1)` 初始化，dummy/padding token 会被 KV 写入 kernel 跳过，不再向 slot 0 写入临时 K/V；`context_lens` 初始化为 0，因此 padding decode 也不会读取历史缓存。

## Step 6：LLM Engine

实现：[llm_engine.py](../src/minivllm/engine/llm_engine.py)、[llm.py](../src/minivllm/llm.py)

`LLM` 当前是 `LLMEngine` 的空子类，直接继承其实现并作为轻量入口。初始化顺序为：

1. 构造 `Config`，设置全局 `Sequence.block_size`。
2. 使用 spawn 创建 rank 1 到 `world_size - 1` 的 worker。
3. 在主进程创建 rank 0 ModelRunner。
4. 加载 tokenizer 并写入 EOS token id。
5. ModelRunner 已计算 `max_cache_blocks` 后，再创建 Scheduler。

Scheduler 必须后创建，因为它需要 ModelRunner 根据实际显存得到的 `config.max_cache_blocks`。

`generate()` 接受字符串 prompt 或 token id 列表，也支持一组独立的 `SamplingParams`。循环调用 `step()` 直到 waiting/running 都为空，最后按 `seq_id` 恢复输入顺序并返回：

```python
{"text": decoded_text, "token_ids": completion_token_ids}
```

`use_tqdm=False` 可关闭生成进度条，适合 benchmark。

## Step 7：Benchmark

### 7.1 Prefill Attention

文件：[benchmark_prefilling.py](../benchmarks/benchmark_prefilling.py)

```bash
python3 benchmarks/benchmark_prefilling.py
```

比较三项：CPU PyTorch FP32、GPU PyTorch FP16、自定义 GPU Triton FlashAttention FP16。输入构造和 CPU/GPU 传输不计时，长序列会减少 CPU 重复次数。正确性同时报告 Triton 相对 CPU FP32 和相对 GPU PyTorch FP16 的最大绝对误差；评价 kernel 本身应重点看后者。

调用 Triton wrapper 时显式传入 `max_seqlen_q`，避免计时循环中通过 `.item()` 产生 GPU/CPU 同步。

### 7.2 Decode PagedAttention

文件：[benchmark_decoding.py](../benchmarks/benchmark_decoding.py)

```bash
python3 benchmarks/benchmark_decoding.py
```

比较 CPU PyTorch FP32、GPU PyTorch FP16 和自定义 GPU Triton PagedAttention FP16。PyTorch baseline 会根据 block table 重建 padded K/V；Triton 直接读取分页缓存，因此这个对比同时包含“重建连续 K/V”和“直接分页访问”的实现差异。

`context_lens_host` 和 `max_context_len` 在计时区外准备，避免 GPU baseline 循环中的 `.item()` 或动态布尔索引同步。

### 7.3 端到端 TPS

文件：[benchmark_tps.py](../benchmarks/benchmark_tps.py)

三个后端应在独立进程中分别启动，避免前一个引擎残留的显存和运行时状态影响后一个。下面给出一组固定长度测试命令：

```bash
python3 benchmarks/benchmark_tps.py \
  --backend minivllm \
  --max-input-tokens 128 \
  --num-sequences 32 \
  --max-output-tokens 256 \
  --warmup-steps 1 --repeat 3 --seed 0 \
  --model-dtype float16 --gpu-memory-utilization 0.9

VLLM_USE_V1=0 python3 benchmarks/benchmark_tps.py \
  --backend vllm \
  --max-input-tokens 128 \
  --num-sequences 32 \
  --max-output-tokens 256 \
  --warmup-steps 1 --repeat 3 --seed 0 \
  --model-dtype float16 --gpu-memory-utilization 0.9

python3 benchmarks/benchmark_tps.py \
  --backend transformers \
  --max-input-tokens 128 \
  --num-sequences 32 \
  --max-output-tokens 256 \
  --warmup-steps 1 --repeat 3 --seed 0 \
  --model-dtype float16
```

输入由词表中的随机 token ID 构造；warmup 与计时阶段使用不同的 prompt batch，避免前缀缓存命中抬高 TPS。默认使用固定输入长度和固定 `max_tokens`。对 mini-vLLM 或 vLLM 增加 `--random-length` 后，每条请求的输入长度与 `max_tokens` 会分别在 `[ceil(max/8), max]` 中独立随机采样；Transformers backend 会忽略该选项。

TPS 使用聚合口径：

```text
TPS = 所有 repeat 的总生成 token 数 / 所有 repeat 的总延迟
```

`--respect-eos` 会允许各后端遇到 EOS 提前停止；默认忽略 EOS，更适合固定输出长度的吞吐量比较。

### 7.4 INT8 KV Cache

文件：[benchmark_int8.py](../benchmarks/benchmark_int8.py)

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

FP16、BF16 和 INT8 三种模式分别在独立子进程中运行，避免模型、NCCL、CUDA Graph 与 KV Cache 预分配相互影响。INT8 模式保持模型和激活 dtype 不变，只量化 KV Cache；脚本同时报告吞吐、缓存容量和简短生成质量指标。完整环境、命令和结果见 [Benchmark 详细记录](benchmark.md)。

### 7.5 单元测试

```bash
python3 -m pytest tests -v
```

当前测试套件共 159 项，结果全部通过。CUDA 可用时，`test_attention.py` 会实际运行 INT8 KV Cache 写入、decode 和 chunked prefill Triton kernel，并与参考结果比较。

## 推荐学习顺序

1. `activation.py`、`rmsnorm.py`：理解 Transformer 的逐元素算子与残差。
2. `linear.py`、`embedding_head.py`：理解张量并行和权重分片。
3. `rotary_embedding.py`、`attention.py`：理解位置编码、Paged KV Cache、INT8 量化/反量化和 Triton Attention。
4. `model_factory.py`、`qwen3.py`、`llama.py`：理解模型创建、运行时元数据和基础层组合。
5. `sequence.py`、`block_manager.py`：理解请求状态和 KV Cache 生命周期。
6. `scheduler.py`：理解 prefix cache、chunked prefill、decode 和抢占。
7. `model_runner.py`：串联输入准备、多卡执行代码路径、不同 dtype 的缓存分配、显存估算和 Decode CUDA Graph Replay。
8. `llm_engine.py`：理解对外 API 与完整生成循环。
9. 四个 benchmark：分别观察 prefill、decode、端到端吞吐，以及 INT8 KV Cache 的性能、容量和生成质量。

## 进一步练习

1. 为 prefill/decode benchmark 增加非等长序列和随机 block table，覆盖真正的 varlen/paged 场景。
2. 优化长序列 prefill 的 `BLOCK_M/BLOCK_N`、`num_warps`、访存和流水线。
3. 调优 decode kernel 的 `BLOCK_N`、`num_warps`、寄存器占用和分页访存，并覆盖不等长 context 与随机 block table。
4. 在不同上下文长度和模型上评估 INT8 KV Cache 的显存节省、吞吐量与输出误差。
5. 扩展采样参数，加入 greedy、top-k、top-p 和 repetition penalty。
6. 改进 Scheduler，使多个请求可以在同一轮共享 chunked prefill 预算。
7. 将硬编码的 `_MODEL_BUILDERS` 扩展为基于模型 architecture/model type 的别名表、注册机制或插件机制。
8. 先在真实多 GPU 环境验证 TP 的正确性和吞吐，再评估自定义 all-reduce、通信计算重叠及不同 `world_size` 下的性能。
