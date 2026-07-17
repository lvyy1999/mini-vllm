# 权重加载单元测试

## 运行命令

```bash
python3 -m pytest tests/test_loader.py -v
```

测试通过伪造 `safe_open` 返回值验证加载流程，不下载模型，也不需要真实 checkpoint 权重。

## 覆盖范围

### TestDefaultWeightLoader

- 形状一致时完整复制权重。
- 形状不一致时抛出带尺寸信息的异常，并保持原参数不变。

### TestCheckpointResolution

- 本地目录没有 `.safetensors` 文件时给出明确错误。
- Hugging Face 下载失败时保留模型名和原始异常原因。

### TestCheckpointLoading

- checkpoint 分片按文件名排序读取。
- 每个权重只调用一次 `get_tensor`，防止重复重建大张量。
- QKV 等 packed 权重按映射交给参数自定义 loader。
- 加载结束后报告未出现在 checkpoint 中的模型参数。
