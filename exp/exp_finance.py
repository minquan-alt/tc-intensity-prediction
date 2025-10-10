import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim

from utils.model_utils import train_val, test_rse_corr
from utils.load_dataset import DataModule
from model.rnn_kan_v1_1 import RNN_KAN

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

    
if __name__ == "__main__":
    file_paths = [
    'artifacts/raw_electricity-v0/raw_electricity.csv',
    'artifacts/raw_exchange_rate-v0/raw_exchange_rate.csv',
    ]
    #====================== Train model ======================#
    # config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {torch.cuda.get_device_name(0)} for device")

    # hyperparams
    in_features = (20, 50, 100)
    hidden_features = 100

    window_size = 168
    n_aheads = [3, 6, 12, 24]
    epochs = 10
    num_workers = 0
    pin_memory = True

    results = {}
    for file_path in file_paths:
        if file_path == 'artifacts/raw_electricity-v0/raw_electricity.csv':
            batch_size = 32
        if file_path == 'artifacts/raw_electricity-v0/raw_exchange_rate.csv':
            batch_size = 4
        for n_ahead in n_aheads:
            print(f"================== TRAINING n_ahead = {n_ahead} ==================")
            dm = DataModule(
            file_path=file_path,
            )

            dm.setup()
            dm.scale()
            train_loader, val_loader, test_loader = dm.get_dataloader(batch_size=batch_size, window_size=window_size, horizon=n_ahead)
            sample_x, sample_y = dm.train_dataset[0]

            in_features = (sample_x.shape[-1], 50, 100)
            out_features = sample_y.shape[-1]

            model = RNN_KAN(in_features, hidden_features, out_features, n_ahead)
            model = model.to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            model, total_time, train_loss_per_epoch, val_loss_per_epoch = train_val(model, criterion, optimizer, train_loader, val_loader, device, batch_size, epochs)

            mae, mse = test_rse_corr(model, test_loader, dm, device)

            results[file_path][n_ahead] = {
                "RSE": mae,
                "CORR": mse,
                "time": [total_time]
            }
    
    print(results)