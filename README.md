# sysid-pytorch-lru

A PyTorch implementation of DeepMind's [Linear Recurrent Unit](https://arxiv.org/pdf/2303.06349) (LRU). Application in System Identification included as example.

## LRU block
The LRU Layer is a linear dynamical system implemented in state-space form as:
$$
\begin{align}
x_{k} = Ax_{x-1} + B u_k\\
y_k = RE[C x_k] + D u_k,
\end{align}
$$
where $A$ is diagonal and complex-valued; $B, C$ are full complex-valued; $D$ is full real-valued. 

## Basic usage:
The basic usage of the LRU block is illustrated in [playground.ipynb](playground.ipynb):

```python
import torch
from lru.linear import LRU

d_state = 200  # state dimension (x)
d_in = 100 # input dimension (u)
d_out = 10 # output dimension (y)
seq_len = 10000  # input sequence length
batch_size = 32

lru = LRU(
    in_features=d_in,
    out_features=d_out,
    state_features=d_state,
)

input_sequences = torch.randn((batch_size, seq_len, d_in))
x0 = torch.view_as_complex(
    torch.randn(batch_size, d_state, 2)
)

# slow loop implementation
output_sequences_loop = lru(input_sequences, mode="loop", state=x0)

# fast parallel scan implementation
output_sequences_scan = lru(input_sequences, mode="scan", state=x0)
```

## Example
System identification of the [Wiener-Hammerstein Benchmark](https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein), see files [train.py](train.py) and [test.ipynb](test.ipynb).
