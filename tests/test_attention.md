# Attention 单元测试

## 运行命令

```bash
python3 -m pytest tests/test_attention.py -v
```

控制流测试使用 mock 验证 `Attention.forward` 的参数分派，不会启动 Triton kernel。INT8 内核测试会在 CUDA 设备上实际运行量化写入、Decode 和 Chunked Prefill kernel；没有可用 CUDA 设备时，这组用例会自动跳过。

## 覆盖范围

### TestAttentionPrefillDispatch

- 普通 prefill 使用本轮生成的 K/V。
- 已分配 KV Cache 且存在 slot mapping 时，先写入本轮 K/V。
- Chunked prefill 改为从 paged KV Cache 读取完整 K/V，并传递块表和序列长度元数据。

### TestAttentionDecodeDispatch

- Decode 使用缓存 K/V、context length 与 block table。
- 有效缓存会写入新 token 的 K/V；空缓存则跳过写入。

### TestAttentionInt8Dispatch

- INT8 Chunked Prefill 将 K/V scale cache 同时传给缓存写入和 attention kernel。
- INT8 Decode 将 K/V scale cache 同时传给缓存写入和 attention kernel。

### TestInt8KVCacheKernels

- 验证非连续 K/V 输入会按 token、KV head 独立执行对称 INT8 量化，并写入正确的物理槽位。
- 验证正负数舍入、零向量 scale 和未使用槽位保持不变。
- Decode 使用乱序物理块表时，INT8 缓存结果与显式反量化后的 FP16 缓存结果一致。
- Chunked Prefill 使用不同查询/缓存长度和乱序物理块表时，INT8 缓存结果与显式反量化后的 FP16 缓存结果一致。
