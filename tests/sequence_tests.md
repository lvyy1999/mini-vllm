# 序列单元测试

## 运行命令

```bash
python3 -m pytest tests/test_sequence.py -v
```

## 覆盖范围

### TestSequenceInitialization

- 验证 prompt token 会被复制，外部列表修改不会污染序列状态。
- 检查初始计数器、状态、末尾 token 和采样参数。
- 确认序列 ID 唯一且单调递增。

### TestSequenceBlocks

- 使用大小为 4 的测试块检查完整块与尾部非完整块的切片。
- 覆盖序列长度是及不是块大小整数倍的情况。
- 拒绝负数和越界块索引。

### TestSequenceMutation

- 验证追加 token 后的总长度、末尾 token 和 completion token 视图。
- 检查 `is_finished` 与序列状态是否一致。

### TestSequenceTransferState

- Prefill 的传输状态包含完整 token 列表。
- Decode 的传输状态只携带最后一个 token，同时保留模型 worker 所需的计数器和块表。
