# 采样参数单元测试

## 运行命令

```bash
python3 -m pytest tests/test_sampling_parameters.py -v
```

## 覆盖范围

- 检查 `temperature`、`max_tokens` 和 `ignore_eos` 的默认值。
- 验证自定义参数能够被完整保留。
- 接受高于实现阈值的正温度。
- 拒绝负数、零以及会退化为 greedy sampling 的温度。
