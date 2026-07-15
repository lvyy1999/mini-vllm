import torch


class RMSNorm(torch.nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # Use nn.Parameter to make weight(gamma) learnable and loadable from checkpoints
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms * self.weight
        return x_norm

    @torch.compile
    def residual_rms_forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + residual
        return self.rms_forward(x), x

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)
