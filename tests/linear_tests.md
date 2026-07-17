# 张量并行线性层单元测试

## 运行命令

```bash
python3 -m pytest tests/test_linear.py -v
```

测试通过 mock `torch.distributed` 的 rank、world size 和通信函数，在单进程 CPU 环境检查切分逻辑，不需要启动 NCCL 进程组。

## 覆盖范围

### TestReplicatedLinear

- 加载完整权重与偏置，并对照 `torch.nn.functional.linear`。
- 验证关闭偏置的参数状态。

### TestColumnParallelLinear

- 检查不同 rank 沿输出维度加载对应分片。
- 验证前向只返回本 rank 的局部输出。
- 拒绝不能被张量并行规模整除的输出维度。

### TestMergedColumnParallelLinear

- 验证多个矩阵合并后，各自的本地分片会写入正确偏移。

### TestQKVColumnParallelLinear

- 分别检查 Q、K、V 的分片大小和装载区间。
- 拒绝未知的权重标识符。

### TestRowParallelLinear

- 检查沿输入维度的权重切分。
- 验证非零 rank 不重复添加偏置，并调用 `all_reduce` 汇总局部结果。
- 验证 rank 0 的偏置路径和输入维度整除约束。
