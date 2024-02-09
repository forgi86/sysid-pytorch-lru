# sysid-pytorch-lru

A PyTorch implementation of DeepMind's [Linear Recurrent Unit](https://arxiv.org/pdf/2303.06349) (LRU). Small working example: system identification of the [Wiener-Hammerstein Benchmark](https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein)

## Usage:
The basic usage of the LRU block is shown in playground.ipynb

    from lru.linear import LRU

    d_state = 200  # hidden state dimension
    d_in = 100
    d_out = 10
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

    d_state = 200 # hidden state dimension
    d_in = 100
    d_out = 10
    seq_len = 10000 # input sequence length
    batch_size = 32

    # fast parallel scan implementation
    output_sequences_scan = lru(input_sequences, mode="scan", state=x0)