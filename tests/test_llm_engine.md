# LLM Engine 单元测试

## 运行命令

```bash
python3 -m pytest tests/test_llm_engine.py -v
```

测试通过 `LLMEngine.__new__` 跳过模型和 GPU 初始化，并使用 mock 隔离 tokenizer、scheduler、model runner 与 worker 进程。

## 覆盖范围

### TestEnginePromptHandling

- 字符串 prompt 先经过 tokenizer，再转换为 `Sequence`。
- token ID 列表绕过 tokenizer，并由序列复制保存。
- 引擎完成状态直接委托给 scheduler。

### TestEngineStep

- 验证 `schedule -> model_runner.run -> postprocess` 调用链。
- 只返回本轮已经结束的序列及其 completion token。

### TestEngineGenerate

- 标量采样参数会用于所有 prompt，列表参数则逐项配对。
- 即使完成顺序不同，最终结果仍按序列 ID 排序并解码。

### TestEngineCleanup

- 退出时通知主 model runner，并等待全部 worker 进程结束。
