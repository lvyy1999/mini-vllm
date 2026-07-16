import torch
import torch.nn as nn


class SamplerLayer(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(
        self,
        logits: torch.Tensor,  # shape(batch_size, vocab_size)
        temperature: torch.Tensor,  # shape(batch_size, )
    ) -> torch.Tensor:
        logits = logits.float() / temperature.unsqueeze(-1)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
