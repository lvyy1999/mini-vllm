# 全局上下文单元测试

## 运行命令

```bash
python3 -m pytest tests/test_context.py -v
```

## 覆盖范围

- 检查默认上下文中的 prefill 标记、序列长度和张量字段。
- 验证 `get_context` 返回当前全局实例。
- 验证 `set_context` 能完整保存 prefill/decode 所需的张量与标量。
- 再次设置上下文时，未传入字段应恢复默认值而不是沿用旧数据。
- 验证 `reset_context` 会创建新的干净上下文。
