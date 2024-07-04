import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-8, bias: bool = False) -> None:
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.scale)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        var = input.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = input * torch.rsqrt(var)

        rmsnorm = self.scale * input_norm

        if self.bias is not None:
            rmsnorm = rmsnorm + self.bias

        return rmsnorm
