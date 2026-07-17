# 调度器单元测试

## 环境准备

```bash
python3 -m pip install pytest numpy xxhash
```

项目本身依赖 PyTorch；应先按项目根目录 `README.md` 完成运行环境安装。

## 运行命令

运行调度器的全部测试：

```bash
python3 -m pytest tests/test_scheduler.py -v
```

按测试类运行：

```bash
python3 -m pytest tests/test_scheduler.py::TestBug2TokenLimitBreak -v
python3 -m pytest tests/test_scheduler.py::TestBug1CanAppendFailure -v
python3 -m pytest tests/test_scheduler.py::TestSchedulerHappyPath -v
python3 -m pytest tests/test_scheduler.py::TestSchedulerQueueState -v
python3 -m pytest tests/test_scheduler.py::TestChunkedPrefill -v
python3 -m pytest tests/test_scheduler.py::TestSchedulerPostprocess -v
```

## 覆盖范围

### TestBug2TokenLimitBreak

验证 decode 循环在 token 预算或序列数上限耗尽时，不会把尚未调度的序列静默丢失。

- `test_seq_count_is_correct`：2-token 预算只能容纳两个序列，`seq_c` 必须保留在 `running`。
- `test_seq_count_limit_variant`：覆盖由 `max_num_sequences` 触发的同类边界。
- `test_no_sequence_is_lost`：所有序列都必须存在于 `running`、`waiting` 或本轮调度结果中。

### TestBug1CanAppendFailure

验证 `block_manager.can_append` 返回 `False` 并触发抢占时，当前序列和队尾序列都不会丢失。

- `test_seq_a_not_lost`：调用结束后仍能在调度器状态中找到 `seq_a`。
- `test_total_conservation`：`seq_a` 与 `seq_b` 均不会消失。

### TestSchedulerHappyPath

覆盖正常路径，包括 prefill 优先、预算充足时的 decode，以及单序列无法扩展时的自抢占。

- 新加入的序列先执行 prefill，完成后进入 `running`。
- 预算充足时，所有运行中序列都会参与本轮 decode。
- 唯一运行序列无法申请新块时，会回到 `waiting` 并恢复为 prefill 状态。

### TestSchedulerQueueState

覆盖空闲状态判断、加入等待队列，以及抢占后状态重置和队首插入顺序。

### TestChunkedPrefill

验证只有批次中的第一个序列可以执行 chunked prefill，并验证下一轮从已缓存位置继续调度。

### TestSchedulerPostprocess

覆盖 chunked prefill 的计数更新、prefill 完成后的首 token、EOS、`ignore_eos` 和 `max_tokens` 终止条件，以及结束序列的块释放。
