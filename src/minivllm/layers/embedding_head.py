import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from minivllm.utils import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # some models' vocab_size is odd, so maybe need to add padding
        # assert vocab_size % self.tp_size == 0, "vocab_size must be divisible by tensor parallel size."
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # pad to make it divisible by tp_size
        self.padded_vocab_size = (vocab_size + self.tp_size - 1) // self.tp_size * self.tp_size
        # this is the vocab_size of per partition in this current GPU
        self.shard_vocab_size = self.padded_vocab_size // self.tp_size

        self.weight = nn.Parameter(torch.empty(self.shard_vocab_size, hidden_size))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        shard_size = self.shard_vocab_size
        start_index = self.tp_rank * shard_size
        actual_size = min(self.vocab_size - start_index, shard_size)

        # load the actual weights
        shard_weights = loaded_weights.narrow(0, start_index, actual_size)
        param_data[:actual_size].copy_(shard_weights)

        # pad the rest with zeros if needed
        if actual_size < shard_size:
            param_data[actual_size:].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mask for tokens in this partition's range and within original vocab size
        mask = (x >= self.tp_rank * self.shard_vocab_size) & \
               (x < (self.tp_rank + 1) * self.shard_vocab_size) & \
               (x < self.vocab_size)
        x = mask * (x - self.tp_rank * self.shard_vocab_size)
        output = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # need to mask again, otherwise the embedding for the out-of-range ids will be the embedding of id 0
            output = mask.unsqueeze(1) * output
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output


# weight tying with embedding layer
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int
    ):
        super().__init__(vocab_size, hidden_size)

    # x: [batch_size, seq_len, hidden_size]
    # weight: [shard_vocab_size, hidden_size]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            # cu_seqlens_q = [0, 5, 8, 12]
            # last_indices = [5, 8, 12] - 1 = [4, 7, 11]
            last_token = context.cu_seqlens_q[1:] - 1  # exclude the first element which is 0
            x = x[last_token].contiguous()

        # logits: [batch_size, seq_len, shard_vocab_size]
        # F.linear automatically transpose the weight
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # prepare for all_gather only for GPU 0 which is the main GPU
            all_logits = [
                torch.empty_like(logits)
                for _ in range(self.tp_size)
            ] if self.tp_rank == 0 else None
            # dist.gather collects the logits from all GPUs to GPU 0
            dist.gather(logits, gather_list=all_logits, dst=0)
            # concatenate
            if self.tp_rank == 0:
                # [batch_size, seq_len, padded_vocab_size]
                logits = torch.cat(all_logits, dim=-1)
                # trim to original vocab size
                logits = logits[..., :self.vocab_size]

        return logits