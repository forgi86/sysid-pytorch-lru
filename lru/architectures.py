import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from .linear import LRU


@dataclass
class DWNConfig:
    d_model: int = 10
    d_state: int = 64
    n_layers: int = 6
    dropout: float = 0.0
    bias: bool = True
    rmin: float = 0.0
    rmax: float = 1.0
    max_phase: float = 2*math.pi
    ff: str = "GLU"


class MLP(nn.Module):
    """ Standard Transformer MLP """
    def __init__(self, config: DWNConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GLU(nn.Module):
    """ The static nonlinearity used in the S4 paper"""
    def __init__(self, config: DWNConfig):
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Linear(config.d_model, 2 * config.d_model),#nn.Conv1d(config.d_model, 2 * config.d_model, kernel_size=1),
            nn.GLU(dim=-1),
        )

    def forward(self, x):
        x = self.dropout(self.activation(x))
        x = self.output_linear(x)
        return x
    
class DWNBlock(nn.Module):
    def __init__(self, config: DWNConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)
        self.lru = LRU(config.d_model, config.d_model, config.d_state,
                        rmin=config.rmin, rmax=config.rmax, max_phase=config.max_phase)
        #self.ff = MLP(config)  # feedforward layer, or GLU
        match config.ff:
            case "GLU":
                self.ff = GLU(config)
            case "MLP":
                self.ff = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, state=None, mode="scan"):

        z = x
        z = self.ln(z)  # prenorm

        z = self.lru(z, state, mode)

        z = self.ff(z) # MLP or GLU
        z = self.dropout(z)

        # Residual connection
        x = z + x

        return x


class DWN(nn.Module):
    def __init__(self, n_u, n_y, config: DWNConfig):
        super().__init__()
        self.encoder = nn.Linear(n_u, config.d_model)
        self.blocks = nn.ModuleList(
            [DWNBlock(config) for _ in range(config.n_layers)]
        )
        self.decoder = nn.Linear(config.d_model, n_y)

    def forward(self, u, state=None, mode="scan"):
        x = self.encoder(u)

        for layer, block in enumerate(self.blocks):
            state_block = state[layer] if state is not None else None
            x = block(x, state=state_block, mode=mode)
        x = self.decoder(x)
        return x
