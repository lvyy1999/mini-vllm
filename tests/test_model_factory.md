# 模型工厂单元测试

## 运行命令

```bash
python3 -m pytest tests/test_model_factory.py -v
```

测试将实际模型类替换为只记录参数的轻量对象，因此不会分配模型权重或 GPU 显存。

## 覆盖范围

### TestRuntimeModelInfo

- 验证运行时模型元数据不可变。

### TestQwen3Builder

- 检查 Qwen3 默认结构参数。
- 验证 Hugging Face 配置字段到自定义模型构造参数的映射。

### TestLlamaBuilder

- 检查 Llama 默认结构参数。
- 验证 attention bias 与 MLP bias 等模型特有字段。

### TestCreateModel

- 根据模型路径的最后一段选择构造器，并原样传递配置字典。
- 不支持的模型会给出原始名称及完整支持列表。
