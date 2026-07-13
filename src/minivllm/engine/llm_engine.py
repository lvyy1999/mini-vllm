import atexit
import torch.multiprocessing as mp
from typing import Any
from tqdm.auto import tqdm
from time import perf_counter
from dataclasses import fields
from transformers import AutoTokenizer

from minivllm.utils.config import Config
from minivllm.engine.sequence import Sequence
from minivllm.engine.scheduler import Scheduler
from minivllm.engine.model_runner import ModelRunner
from minivllm.sampling_parameters import SamplingParams


def run_worker(config, rank, event):
    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()

class LLMEngine:
    def __init__(self, **kwargs):
        # load config
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(**config_kwargs)
        # set block size
        Sequence.block_size = config.cache_block_size
        # prepare for multi-gpu if needed
        self.events = []
        self.processes = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.world_size):
            event = ctx.Event()
            # create a process to run worker in other gpu
            process = ctx.Process(target=run_worker, args=(config, i, event))
            process.start()
            self.events.append(event)
            self.processes.append(process)
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        # get tokenizer and set EOS
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        config.eos_token_id = self.tokenizer.eos_token_id
        # create scheduler
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self) -> tuple[list[tuple[Any, Any]], int, bool]:
        # call scheduler to schedule the next batch
        seqs, num_tokens, is_prefill = self.scheduler.schedule()
        # call model_runner.run() to run the model and get output
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # call scheduler to do post-process
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # if seq is finished, add it into output
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens, is_prefill

    # add prompt to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str | list[int], sampling_params: SamplingParams) -> None:
        if isinstance(prompt, str): # if prompt is a string, tokenize it
            prompt = self.tokenizer.encode(prompt)
        self.scheduler.add_sequence(Sequence(token_ids=prompt, sampling_params=sampling_params))

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def generate(
            self,
            prompts: list[str] | list[list[int]],
            sampling_params: SamplingParams | list[SamplingParams],
            use_tqdm: bool = True,
    ) -> list[dict[str, Any]]:
        # create tqdm bar
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # add prompts
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_prompt(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, is_prefill = self.step()
            running_time = perf_counter() - t
            if is_prefill:
                prefill_throughput = num_tokens / running_time
                # print(num_tokens, ' tokens processed, ', prefill_throughput, " tokens/s during prefilling")
            else:
                decode_throughput = num_tokens / running_time
                # print(num_tokens, ' tokens processed, ', decode_throughput, " tokens/s during decoding")
            pbar.set_postfix({
                "Prefill": f"{num_tokens if is_prefill else 0} tokens processed, {int(prefill_throughput)}tokens/s",
                "Decode": f"{num_tokens if not is_prefill else 0} tokens processed, {int(decode_throughput)}tokens/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
