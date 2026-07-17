# 词嵌入与 LM Head 单元测试

## 运行命令

```bash
python3 -m pytest tests/test_embedding_head.py -v
```

测试使用 mock 的分布式接口模拟多个 rank，不需要真正启动 NCCL。

## 覆盖范围

### TestVocabParallelEmbedding

- 单 rank 输出与 `torch.nn.functional.embedding` 一致。
- 词表大小不能整除并行规模时，最后一个 rank 正确加载有效行并将填充行清零。
- 非本 rank token 和超出原始词表的填充 token 会被掩码，并调用 `all_reduce` 汇总结果。

### TestParallelLMHead

- Decode 阶段为所有输入 hidden state 计算 logits。
- Prefill 阶段依据 `cu_seqlens_q` 只选择每个序列的最后一个 hidden state。
- rank 0 收集各分片 logits，并裁掉 padded vocabulary 对应的多余列。
