# KV Cache 块管理器单元测试

## 运行命令

```bash
python3 -m pytest tests/test_block_manager.py -v
```

## 覆盖范围

### TestBlockLifecycle

- 检查空闲块与已用块的初始状态。
- 验证分配、释放、引用计数和序列元数据更新。
- 覆盖容量不足以及从空池申请块的异常路径。

### TestAppendCapacity

- 生成 token 跨越块边界时申请新块。
- 没有空闲块时拒绝跨块扩展。
- 当前块仍有空间时不进行多余分配。

### TestBlockHashing

- 只为已经完整且完成调度的块计算哈希。
- 验证前一个块的哈希会参与下一个块的哈希计算。
- 人为制造哈希碰撞，确认 token ID 比较能够阻止错误命中。

### TestPrefixCacheReuse

- 在活动序列之间共享完整前缀块。
- 原序列结束后，从空闲队列重新启用仍保留缓存信息的块。
- 共享块在最后一个引用释放前保持已分配状态。
- 回归验证最后一个块即使 token 完全相同，也不会在活动序列之间共享。
