# 单元测试说明

## 环境准备

先按项目根目录 `README.md` 安装与 CUDA 环境匹配的 PyTorch 及项目运行依赖，再安装测试工具：

```bash
python3 -m pip install pytest
```

## 完整测试命令

在项目根目录运行：

```bash
python3 -m pytest tests -v
```

只运行不依赖张量计算的核心状态测试：

```bash
python3 -m pytest \
  tests/test_config.py \
  tests/test_sampling_parameters.py \
  tests/test_sequence.py \
  tests/test_block_manager.py \
  tests/test_scheduler.py \
  -v
```

## 测试分层

| 测试文件 | 主要范围 | 是否启动 CUDA kernel |
| --- | --- | --- |
| `test_config.py` | 引擎配置与约束 | 否 |
| `test_sampling_parameters.py` | 采样参数 | 否 |
| `test_sequence.py` | 序列状态、分块与传输状态 | 否 |
| `test_block_manager.py` | 块生命周期与 prefix cache | 否 |
| `test_scheduler.py` | Prefill、decode、抢占与后处理 | 否 |
| `test_context.py` | Attention 全局上下文 | 否 |
| `test_core_layers.py` | 激活、RMSNorm、RoPE 与 sampler | 否 |
| `test_linear.py` | 张量并行线性层 | 否，分布式接口使用 mock |
| `test_embedding_head.py` | 词嵌入与 LM Head | 否，分布式接口使用 mock |
| `test_attention.py` | Attention prefill/decode 分派 | 否，Triton 函数使用 mock |
| `test_loader.py` | checkpoint 与 packed 权重加载 | 否，文件读取使用 fake |
| `test_model_factory.py` | 模型构造参数映射 | 否，模型类使用替身 |
| `test_model_runner.py` | 输入准备、CUDA Graph 缓冲区和 RPC | 否，设备传输与 graph 使用 mock |
| `test_llm_engine.py` | 引擎编排与输出收集 | 否，外部组件使用 mock |

## 范围说明

- `tests/conftest.py` 设置 `TORCHDYNAMO_DISABLE=1`，单元测试关注算子逻辑，不承担 `torch.compile` 性能验证。
- 测试通过源码包命名空间直接导入子模块，避免导入一个轻量模块时由顶层 `minivllm.__init__` 提前初始化完整 CUDA 运行栈。
- Triton attention kernel 的真实数值、显存访问和性能依赖 GPU，仍应使用项目 benchmark 在目标 CUDA 环境验证。
- 每个测试文件都有同名的 `*_tests.md` 中文文档，包含单独运行命令和覆盖说明。
