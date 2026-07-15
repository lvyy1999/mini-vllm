import torch
import pickle
import torch.distributed as dist
from torch import Tensor
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from minivllm.models.qwen3 import Qwen3ForCausalLM
from minivllm.models.llama import LlamaForCausalLM
from minivllm.layers.sampler import SamplerLayer
from minivllm.engine.sequence import Sequence
from minivllm.utils.context import *
from minivllm.utils.loader import load_weights_from_checkpoint
from minivllm.utils.config import Config

dtype_mapping = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.rank = rank
        self.event = event
        self.graphs = {}
        self.graph_vars = {}
        self.max_graph_bs = 512
        self.world_size = config.world_size
        self.block_size = config.cache_block_size
        self.enforce_eager = config.enforce_eager

        # set dist
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # set default device and dtype for model
        self.model_dtype = dtype_mapping[config.custom_model_config.get('torch_dtype', 'float16')]
        default_dtype = torch.get_default_dtype()
        torch.set_default_device(f'cuda:{rank}')
        torch.set_default_dtype(self.model_dtype)

        # create model
        model_name = Path(config.model_name_or_path).name
        match model_name:
            case 'Qwen3-0.6B':
                self.num_layers = config.custom_model_config.get('num_hidden_layers', 28)
                self.num_kv_heads = config.custom_model_config.get('num_key_value_heads', 8)
                self.head_dim = config.custom_model_config.get('head_dim', 128)
                self.hidden_size = config.custom_model_config.get('hidden_size', 1024)
                self.model = Qwen3ForCausalLM(
                    vocab_size=config.custom_model_config.get('vocab_size', 151936),
                    hidden_size=self.hidden_size,
                    num_heads=config.custom_model_config.get('num_attention_heads', 16),
                    head_dim=self.head_dim,
                    num_kv_heads=self.num_kv_heads,
                    max_position=config.custom_model_config.get('max_position_embeddings', 40960),
                    rms_norm_eps=config.custom_model_config.get('rms_norm_eps', 1e-6),
                    intermediate_size=config.custom_model_config.get('intermediate_size', 3072),
                    qkv_bias=config.custom_model_config.get('attention_bias', False),
                    base=config.custom_model_config.get('rope_theta', 1000000.0),
                    num_layers=self.num_layers,
                    tie_word_embeddings=config.custom_model_config.get('tie_word_embeddings', True),
                )
            case 'Llama-3.2-1B-Instruct':
                self.num_layers = config.custom_model_config.get('num_hidden_layers', 16)
                self.num_kv_heads = config.custom_model_config.get('num_key_value_heads', 8)
                self.head_dim = config.custom_model_config.get('head_dim', 64)
                self.hidden_size = config.custom_model_config.get('hidden_size', 2048)
                self.model = LlamaForCausalLM(
                    vocab_size=config.custom_model_config.get('vocab_size', 128256),
                    hidden_size=self.hidden_size,
                    num_heads=config.custom_model_config.get('num_attention_heads', 32),
                    head_dim=self.head_dim,
                    num_kv_heads=self.num_kv_heads,
                    max_position=config.custom_model_config.get('max_position_embeddings', 131072),
                    rms_norm_eps=config.custom_model_config.get('rms_norm_eps', 1e-5),
                    intermediate_size=config.custom_model_config.get('intermediate_size', 8192),
                    qkv_bias=config.custom_model_config.get('attention_bias', False),
                    ffn_bias=config.custom_model_config.get('mlp_bias', False),
                    base=config.custom_model_config.get('rope_theta', 500000.0),
                    num_layers=self.num_layers,
                    tie_word_embeddings=config.custom_model_config.get('tie_word_embeddings', True),
                )
            case _:
                raise Exception(f"Unsupported model: {config.model_name_or_path}")
        # load pretrained model weights
        load_weights_from_checkpoint(self.model, config.model_name_or_path)
        # set sampler
        self.sampler = SamplerLayer()
        # warm up model so that we know peak memory usage
        self.warmup_model()
        # allocate kv cache
        self.allocate_kv_cache()
        # capture cuda graph for decoding
        if not self.enforce_eager:
            self.capture_cudagraph()

        # reset default device and dtype
        torch.set_default_device('cpu')
        torch.set_default_dtype(default_dtype)

        # init shared memory
        if self.world_size > 1:
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                try:
                    old_shm = SharedMemory(name='minivllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, pass
                self.shm = SharedMemory(name='minivllm', create=True, size=2**20)
                # barrier to ensure other rank wait until shared memory is created
                dist.barrier()
            else:
                # wait for rank 0 to create shared memory
                dist.barrier()
                self.shm = SharedMemory(name='minivllm')
                # change to call loop() by outside
                # self.loop()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # Check if process group exists before destroying
        if dist.is_initialized():
            dist.destroy_process_group()

    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)  # Unpack args when calling
            if method_name == 'exit':
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0: # will be called in main engine
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # -> compute max number of sequence based on max token and max model length
    # -> run empty sequence to warm up the model
    # -> cleanup memory
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config.max_num_batched_tokens
        max_model_length = self.config.max_model_length
        seq_len = min(max_tokens, max_model_length)
        batch_size = min(max_tokens // seq_len, self.config.max_num_sequences)
        seqs = [
            Sequence([0] * seq_len) for _ in range(batch_size)
        ]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # find all available memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        avail_free_mem = free_mem * self.config.gpu_memory_utilization
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        # reserve some room for peak memory usage during model execution
        available_mem = avail_free_mem - (peak_mem_usage - current_mem_usage)

        # check whether the current free memory can hold at least one block
        # calculate the actual bytes required of each block
        block_bytes = 2 * self.num_layers * self.block_size * self.num_kv_heads * self.head_dim * self.model_dtype.itemsize
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'

        # Synchronize max_cached_blocks across all ranks.
        # Each rank independently computed num_available_kv_blocks from its own
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead
        # (NCCL buffers, process-group state) so it often has less free memory than
        # workers. Without sync, the scheduler (which runs only on rank-0) would use
        # rank-0's local value and could allocate more blocks than some rank can hold,
        # causing an OOM on that rank during KV cache writes.
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )
            # all_reduce with MIN: every rank learns the most conservative limit,
            # i.e. the block count that even the most memory-constrained rank can serve.
            # This single agreed-upon value is then stored in config so the Scheduler
            # (initialized afterward on rank-0) never allocates more blocks than any
            # rank can physically hold.
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config.max_cache_blocks = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            self.config.max_cache_blocks = num_available_kv_blocks
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config.max_cache_blocks}")

        # allocate max possible kv cache for the model, instead for each sequence
        # this is the key for paged attention: one giant KV cache pool, divided into blocks
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values
        kv_cache = torch.zeros(2, self.num_layers, self.config.max_cache_blocks, self.block_size, self.num_kv_heads, self.head_dim, dtype=self.model_dtype, device=f'cuda:{self.rank}')
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1

    # prepare block tables with padding for chunked prefill or decode
    @classmethod
    def prepare_block_tables(cls, seqs: list[Sequence]):
        max_num_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_num_blocks - len(seq.block_table)) for seq in seqs
        ] # padding with -1
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # prepare the data needed for prefill:
    # input_ids, positions, cu_seqlens_q/k, max_seqlen_q/k,
    # slot_mapping (where to write new KV values),
    # block_tables (where to read KV values).
    # e.g. cu_seqlens_q = [0, 3, 5, 9]
    # seq 0 : 0 - 2, seq 1 : 3 - 4, seq 2 : 5 - 8
    def prepare_prefill(self, seqs: list[Sequence]) -> tuple[Tensor, Tensor]:
        # length: sum of all input_ids after prefix cache
        input_ids = []
        positions = []
        slot_mapping = []
        # length: num_seqs + 1
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        # block_tables: num_seqs x max_num_blocks (padded) or None
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens # start pos in this turn
            seqlen_q = seq.num_scheduled_tokens
            seqlen_k = end = start + seqlen_q # end pos in this turn
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            if not seq.block_table: # warmup
                continue
            # calculate slot for each token
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end % self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]: # chunked prefill
            block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # set global context, used for flash attention
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=block_tables,
        )

        return input_ids, positions

    # prepare the data needed for decode
    def prepare_decode(self, seqs: list[Sequence]) -> tuple[Tensor, Tensor]:
        input_ids = []
        positions = []
        context_lens = []   
        slot_mapping = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # set global context, used for flash attention
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        return input_ids, positions

    # prepare the temperatures for sample
    @classmethod
    def prepare_sample(cls, seqs: list[Sequence]) -> Tensor:
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits;
    # when decoding, use cuda graph execution to speed up
    # if not enforce_eager and batch_size <= max_graph_bs.
    # graph execution: allocate input_ids, positions, slot_mapping, context_lens,
    # block_tables, outputs into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager or input_ids.size(0) > self.max_graph_bs:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0) # batch_size
            context = get_context()
            # finds smallest captured graph that fits the batch size
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            # copy needed data into graph variables
            graph_vars = self.graph_vars
            graph_vars['input_ids'][:bs].copy_(input_ids)
            graph_vars['positions'][:bs].copy_(positions)
            graph_vars['slot_mapping'][:bs].fill_(-1)
            graph_vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            graph_vars["context_lens"].zero_()
            graph_vars['context_lens'][:bs].copy_(context.context_lens)
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # replay the graph
            graph.replay()
            return self.model.compute_logits(graph_vars['outputs'][:bs])

    # prepare prefill/decode
    # -> prepare sample
    # -> run model
    # -> sample logits
    # -> reset context
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        # only sample when rank == 0, and convert them to a list
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated once and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        self.max_graph_bs = min(self.config.max_num_sequences, 512)
        max_len = self.config.max_model_length
        max_num_blocks = (max_len + self.block_size - 1) // self.block_size
        # for decoding, input is always of shape (batch_size, 1)
        input_ids = torch.zeros(self.max_graph_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        positions = torch.zeros(self.max_graph_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.zeros(self.max_graph_bs, dtype=torch.int32, device=f'cuda:{self.rank}')
        # how many tokens each sequence has processed
        context_lens = torch.zeros(self.max_graph_bs, dtype=torch.int32, device=f'cuda:{self.rank}')
        # where to read KV values in the cache
        block_tables = torch.zeros(self.max_graph_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        # output logits
        outputs = torch.zeros(self.max_graph_bs, self.hidden_size, dtype=self.model_dtype, device=f'cuda:{self.rank}')
        # graphs to be captured for different batch sizes
        batch_sizes = [1, 2, 4, 8] + list(range(16, self.max_graph_bs + 1, 16))
        graph_pool = None

        for bs in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            with torch.cuda.graph(graph, graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                if graph_pool is None:
                    graph_pool = graph.pool()
            # store the captured graph
            self.graphs[bs] = graph
            # make sure that the capture is done before resetting and next capture
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
