import math
from argparse import Namespace
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from lru.architectures import DWN, DWNConfig
from tqdm import tqdm
import nonlinear_benchmarks


if __name__ == "__main__":

    # very small architecture
    cfg = {
        "n_u": 1,
        "n_y": 1,
        "d_model": 5,
        "d_state": 5,
        "n_layers": 3,
        "ff": "GLU", # GLU | MLP
        "max_phase": math.pi
    }
    cfg = Namespace(**cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch.set_num_threads(10)

    # Load the benchmark data
    train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
    sampling_time = train_val.sampling_time
    u_train, y_train = train_val   
    u_train = u_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    n_u = 1
    n_y = 1
    
    # Rescale data
    scaler_u = StandardScaler()
    u = scaler_u.fit_transform(u_train)

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y_train)

    # Build model
    config = DWNConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, max_phase=cfg.max_phase)
    model = DWN(cfg.n_u, cfg.n_y, config)

    # Configure optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # Load data
    u = torch.tensor(u).unsqueeze(0).float() # B, T, C
    y = torch.tensor(y).unsqueeze(0).float()

    LOSS = []
    # Train loop
    for itr in tqdm(range(5000)):

        y_pred = model(u)
        loss = torch.nn.functional.mse_loss(y, y_pred)

        loss.backward()
        opt.step()

        opt.zero_grad()
        if itr % 100 == 0:
            print(loss.item())
        LOSS.append(loss.item())

    checkpoint = {
        'scaler_u': scaler_u,
        'scaler_y': scaler_y,
        'model': model.state_dict(),
        'LOSS': np.array(LOSS),
        'cfg': cfg
    }

    torch.save(checkpoint, "ckpt.pt")
