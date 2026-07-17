# 配置单元测试

## 运行命令

```bash
python3 -m pytest tests/test_config.py -v
```

## 覆盖范围

### TestConfigDefaults

- 检查引擎默认配置。
- 验证 `custom_model_config` 使用独立字典，不会在实例之间共享。

### TestWorldSizeValidation

- 接受 1 到 8 范围内的张量并行规模。
- 拒绝 0、负数以及大于 8 的配置。

### TestCacheBlockSizeValidation

- 接受正的 2 的幂次作为 KV Cache 块大小。
- 拒绝非正数和非 2 的幂次。

### TestModelLengthResolution

- 当用户请求长度超过模型上限时，使用 `max_position_embeddings` 截断。
- 用户请求更小时保留用户值。
- 模型配置没有长度字段时不改写用户值。
