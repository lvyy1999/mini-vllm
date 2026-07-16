import torch


# RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ
class RMSNorm(torch.nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # Use nn.Parameter to make weight(gamma) learnable and loadable from checkpoints
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(origin_dtype).mul_(self.weight)
        return x

    @torch.compile
    def residual_rms_forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origin_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(origin_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(origin_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.residual_rms_forward(x, residual)
