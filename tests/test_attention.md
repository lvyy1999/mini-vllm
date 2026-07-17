# Attention 分派单元测试

## 运行命令

```bash
python3 -m pytest tests/test_attention.py -v
```

本文件只验证 `Attention.forward` 的控制流。KV 写入、prefill attention 和 decode attention 均替换为 mock，不会启动 Triton kernel；内核数值正确性应由独立 GPU benchmark 验证。

## 覆盖范围

### TestAttentionPrefillDispatch

- 普通 prefill 使用本轮生成的 K/V。
- 已分配 KV Cache 且存在 slot mapping 时，先写入本轮 K/V。
- Chunked prefill 改为从 paged KV Cache 读取完整 K/V，并传递块表和序列长度元数据。

### TestAttentionDecodeDispatch

- Decode 使用缓存 K/V、context length 与 block table。
- 有效缓存会写入新 token 的 K/V；空缓存则跳过写入。
