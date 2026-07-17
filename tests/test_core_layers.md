# 基础算子单元测试

## 运行命令

```bash
python3 -m pytest tests/test_core_layers.py -v
```

这些测试在 CPU 上比较数学结果。`tests/conftest.py` 会设置 `TORCHDYNAMO_DISABLE=1`，避免 `torch.compile` 的首次编译时间影响单元测试速度；CUDA/Triton attention 内核不在本文件范围内。

## 覆盖范围

### TestSiLUAndMul

- 对照 `silu(gate) * value` 参考公式检查门控激活结果与输出形状。

### TestRMSNorm

- 对照 FP32 参考公式检查 RMSNorm。
- 验证低精度输入的输出 dtype。
- 检查融合 residual 路径返回的残差和归一化结果。

### TestRotaryFunctions

- 分别验证相邻元素配对和前后半区配对的旋转公式。
- 检查位置 0 的恒等性质与 cos/sin 缓存形状。
- 对照缓存手工计算 RoPE，并检查 Llama 3 缩放分支不会产生非有限值。

### TestSamplerLayer

- 使用概率高度集中的 logits 验证逐行采样结果。
- 验证输出形状、整数 dtype 和词表索引范围。
