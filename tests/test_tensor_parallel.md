# 双卡张量并行正确性测试

## 运行命令

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m pytest tests/test_tensor_parallel.py -v -s
```

测试代码会自行使用 `spawn` 启动两个子进程，并分别绑定 `cuda:0` 和 `cuda:1`，因此不要再使用 `torchrun`。每个用例通过本机临时 TCP 端口初始化一个真实 NCCL process group，默认通信超时为 90 秒。

运行条件：

- 至少有两张可见的 CUDA GPU。
- 当前 PyTorch 构建包含 NCCL。
- 已安装 Triton。

条件不满足时，两项测试会自动标记为 `SKIPPED`，不会退化为 mock 通信。

## 实测结果

两项双卡 TP 测试已在真实双 GPU CUDA/NCCL 环境中通过，并随完整测试套件取得 `161 passed`。

## 覆盖范围

### TestTwoGpuTensorParallel.test_tp_layers_match_dense_references

- `ColumnParallelLinear`：两个 rank 分别加载输出分片，通过真实 `all_gather` 重建输出，并与完整 `torch.nn.functional.linear` 结果比较。
- `RowParallelLinear`：两个 rank 分别处理输入分片，通过真实 `all_reduce` 汇总结果，并验证 bias 只累加一次。
- `Qwen3MLP`：覆盖 Gate/Up 合并列并行、SwiGLU 和 Down 行并行的完整前向，并与未切分的稠密参考实现比较。
- `VocabParallelEmbedding`：覆盖不能整除并行规模的词表、末尾 padding 和真实 `all_reduce`。
- `ParallelLMHead`：在 Decode 和 Prefill 两种上下文中通过真实 `gather` 收集词表 logits，由 rank 0 与完整词表参考结果比较。

上述 FP32 路径使用 `rtol=1e-5`、`atol=1e-5`。

### TestTwoGpuTensorParallel.test_qwen_attention_matches_dense_reference

- 使用两个 rank 分片加载 Q/K/V Projection 和 Output Projection，并复制 Q/K RMSNorm 参数。
- 实际执行 Q/K RMSNorm、RoPE、GQA、Triton causal Prefill Attention 和 Output Projection 的双卡 TP 前向。
- 通过 `RowParallelLinear` 的真实 NCCL `all_reduce` 恢复完整 hidden state。
- 与未切分的 PyTorch causal GQA 参考实现比较，并检查输出不存在 NaN 或 Inf。

Attention 使用 FP16，考虑 Triton Online Softmax、矩阵乘法和 NCCL 归约的舍入差异，比较容差为 `rtol=3e-2`、`atol=3e-2`。

## 测试边界

该文件只验证双卡 TP 的数值与通信正确性，不统计延迟或吞吐，也不会下载模型权重。测试使用小尺寸确定性张量，因此不覆盖完整 `LLMEngine`、权重文件加载、Decode CUDA Graph、Prefix Cache、Chunked Prefill 或 INT8 KV Cache。Qwen3-0.6B 的 eager 双卡端到端生成已通过 `main.py` 独立验证，其余场景仍属于后续多卡集成验证范围。
