import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None,
    ):
        super().__init__()
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # create weight parameter with weight loader
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        # create bias parameter
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# the simplest Linear layer: ReplicatedLinear(LinearBase)
# where we simply copy the weight as the weight_loader
# and run the forward as a normal linear layer
class ReplicatedLinear(LinearBase):

    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param.data.copy_(loaded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# column-split Linear layer: ColumnParallelLinear(LinearBase)
# get the original full parameter
# compute the starting index of the column split
# compute the dim size of the full parameter
# copy the parameter slice to the local parameter
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert (
            output_size % tp_size == 0
        ), "Output size must be divisible by tensor parallel size."
        super().__init__(input_size, output_size // tp_size, bias, tp_dim=0)

    # param: parameter after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        # full_dim on the output column
        full_output_size = loaded_weights.size(0)
        # dim size after sharding
        shard_size = full_output_size // self.tp_size
        assert shard_size == param_data.size(
            0
        ), "Shard size does not match parameter size."
        # split the original full weights and copy
        start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(shard_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# an extension of ColumnParallelLinear by merging several matrices
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[
            int
        ],  # e.g. merge QKV matrices to compute MM together and then split
        bias: bool = True,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    # split each weight matrix into tp_size gpus
    # param: parameter to be reloaded after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    # loaded_weight_id: the index of merged matrices (e.g. it's 0 for gate_proj, 1 for up_proj assuming gate_proj and up_proj are merged together)
    def weight_loader(
        self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int
    ):
        param_data = param.data
        # compute offset of param_data
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        # compute shard_size of loaded_weight_id^th matrix
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        # find the correct slice to be loaded in the sharded parameter
        param_data = param_data.narrow(0, offset, shard_size)
        # split the original full weights and copy
        start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim
        assert (
            num_heads % tp_size == 0
        ), "num_heads must be divisible by tensor parallel size."
        self.num_heads = num_heads // tp_size
        assert (
            num_kv_heads % tp_size == 0
        ), "num_kv_heads must be divisible by tensor parallel size."
        self.num_kv_heads = num_kv_heads // tp_size
        output_size = head_dim * (num_heads + 2 * num_kv_heads)
        super().__init__(hidden_size, output_size, bias=bias)

    # load_weight_id: q, k, v
    def weight_loader(
        self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str
    ):
        param_data = param.data
        assert load_weight_id in [
            "q",
            "k",
            "v",
        ], "load_weight_id must be one of 'q', 'k', 'v'"
        # calculate offset and shard_size
        if load_weight_id == "q":
            offset = 0
            shard_size = self.head_dim * self.num_heads
        elif load_weight_id == "k":
            offset = self.head_dim * self.num_heads
            shard_size = self.head_dim * self.num_kv_heads
        elif load_weight_id == "v":
            offset = self.head_dim * self.num_heads + self.head_dim * self.num_kv_heads
            shard_size = self.head_dim * self.num_kv_heads
        # find the correct slice to be loaded in the sharded parameter
        param_data = param_data.narrow(0, offset, shard_size)
        # split the original full weights and copy
        start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(shard_weights)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert (
            input_size % tp_size == 0
        ), "Input size must be divisible by tensor parallel size."
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        # full_dim on the input row
        full_input_size = loaded_weights.size(1)
        # dim size after sharding
        shard_size = full_input_size // self.tp_size
        assert shard_size == param_data.size(
            1
        ), "Shard size does not match parameter size."
        # split the original full weights and copy
        start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(shard_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y
