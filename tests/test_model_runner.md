# Model Runner 单元测试

## 运行命令

```bash
python3 -m pytest tests/test_model_runner.py -v
```

测试不会调用 `ModelRunner.__init__`，因此不会初始化 NCCL、加载模型或申请显存。输入准备测试会将 `.cuda()` 临时替换为 CPU 原样返回，用于验证生成的数据内容。

## 覆盖范围

### TestSharedMemoryRPC

- 验证 rank 0 写入、worker 读取的共享内存序列化往返。
- 检查读写 rank 约束、事件通知、未知方法错误和 worker 的退出条件。
- 验证多卡主进程先广播调用，再执行本地方法。

### TestInputPreparation

- 块表使用 `-1` 对齐到相同列数。
- Prefill 正确生成 token、position、累计序列长度、slot mapping 和 chunked prefill 块表。
- Decode 正确生成最后一个 token、上下文长度、写入槽位和块表。
- 各序列 temperature 按 FP32 张量传入 sampler。

### TestModelExecution

- Prefill 使用 eager 前向路径。
- Decode 选择能容纳当前 batch 的最小 CUDA Graph，并正确填充 padding 和上下文缓冲区。

### TestRunOrchestration

- rank 0 完成输入准备、模型执行、采样并重置全局上下文。
- 非零 rank 只执行模型，不重复采样。
