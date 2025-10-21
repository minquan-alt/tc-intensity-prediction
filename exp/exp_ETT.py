import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

from model.rnn_kan_v1_1 import RNN_KAN


from utils.model_utils import train_val, test_mae_mse
from utils.load_dataset import DataModule

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

if __name__ == "__main__":
    file_paths = [
        'artifacts/raw_ETTh1-v0/raw_ETTh1.csv',
        'artifacts/raw_ETTh2-v0/raw_ETTh2.csv',
        'artifacts/raw_ETTm1-v0/raw_ETTm1.csv',
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.get_device_name(0)} for device")
    else:
        print("Using CPU")

    in_features = (20, 50, 100)
    hidden_features = 100

    epochs = 10
    num_workers = 0
    pin_memory = True

    results = {}
    for file_path in file_paths:
        if file_path == 'artifacts/raw_ETTh1-v0/raw_ETTh1.csv':
            n_aheads = [24, 48, 168, 336, 720]
            window_sizes = [48, 96, 336, 336, 736]
            batch_sizes = [8, 16, 32, 256, 128]
        elif file_path == 'artifacts/raw_ETTh2-v0/raw_ETTh2.csv':
            n_aheads = [24, 48, 168, 336, 720]
            window_sizes = [48, 96, 336, 336, 736]
            batch_sizes = [16, 4, 16, 128, 128]
        elif file_path == 'artifacts/raw_ETTm1-v0/raw_ETTm1.csv':
            n_aheads = [24, 48, 96, 288, 672]
            window_sizes = [48, 96, 384, 672, 672]
            batch_sizes = [32, 16, 32, 32, 32]
        else:
            print("no file found")
        for i in range(len(n_aheads)):
            n_ahead = n_aheads[i]
            window_size = window_sizes[i]
            batch_size = batch_sizes[i]
            print(f"================== TRAINING n_ahead = {n_ahead} ==================")
            # prepare dataset
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

            # metrics = [nn.L1Loss()]
            mae, mse = test_mae_mse(model, test_loader, dm, device)

            results[file_path][n_ahead] = {
                "MAE": mae,
                "MSE": mse,
                "time": [total_time]
            }

    print(results)

