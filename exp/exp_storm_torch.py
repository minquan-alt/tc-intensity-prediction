import numpy as np
import os
import wandb
from collections import defaultdict

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

from utils.data_utils import prepare_data, load_and_process_data
from utils.model_utils import train_val, test
from utils.data_utils import StormDataset, prepare_data
from model import rnn_kan_v1_1
from model import rnn_kan_v1_2 

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



data_path = 'data/cma-era5-data'
raw_train_data, raw_val_data, raw_test_data = load_and_process_data(data_path=data_path, train_years=list(range(1980, 2017)), val_years=[2017, 2018, 2019], test_years=[2020, 2021, 2022])

#====================== Train model ======================#
# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.get_device_name(0)} for device")

# hyperparams
seq_len = 5
n_aheads = list(range(1, 13))
batch_size = 128
epochs = 10
num_workers = 0
lr = 1e-3
pin_memory = True
criterion = nn.MSELoss()


model_ids = ['rnn_kan_v1_1', 'rnn_kan_v1_2']
data_id = 'raw_cma_era5-v0'

results = defaultdict(dict)

for n_ahead in n_aheads:
  print(f"================== TRAINING n_ahead = {n_ahead} ==================")
  X_train, y_train, metadata_train = prepare_data(raw_train_data, sequence_length = seq_len, n_ahead = n_ahead)
  X_val, y_val, metadata_val = prepare_data(raw_val_data, sequence_length = seq_len, n_ahead = n_ahead)
  X_test, y_test, metadata_test = prepare_data(raw_test_data, sequence_length = seq_len, n_ahead = n_ahead)

  print(f"X_shape: {X_train.shape}")
  print(f"y_shape: {y_train.shape}")

  raw_test_data_2020 = {k: v for k, v in raw_test_data.items() if int(k[-4:]) == 2020}
  raw_test_data_2021 = {k: v for k, v in raw_test_data.items() if int(k[-4:]) == 2021}
  raw_test_data_2022 = {k: v for k, v in raw_test_data.items() if int(k[-4:]) == 2022}

  X_test_2020, y_test_2020, metadata_test_2020 = prepare_data(raw_test_data_2020, sequence_length = seq_len, n_ahead = n_ahead)
  X_test_2021, y_test_2021, metadata_test_2021 = prepare_data(raw_test_data_2021, sequence_length = seq_len, n_ahead = n_ahead)
  X_test_2022, y_test_2022, metadata_test_2022 = prepare_data(raw_test_data_2022, sequence_length = seq_len, n_ahead = n_ahead)

  scaler_X = StandardScaler()
  X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
  X_val_scaled   = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
  X_test_scaled  = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
  X_test_2020_scaled = scaler_X.transform(X_test_2020.reshape(-1, X_test_2020.shape[-1])).reshape(X_test_2020.shape)
  X_test_2021_scaled = scaler_X.transform(X_test_2021.reshape(-1, X_test_2021.shape[-1])).reshape(X_test_2021.shape)
  X_test_2022_scaled = scaler_X.transform(X_test_2022.reshape(-1, X_test_2022.shape[-1])).reshape(X_test_2022.shape)

  scaler_y = StandardScaler()
  y_train_scaled = scaler_y.fit_transform(y_train)
  y_val_scaled   = scaler_y.transform(y_val)
  y_test_scaled  = scaler_y.transform(y_test)
  y_test_2020_scaled = scaler_y.transform(y_test_2020)
  y_test_2021_scaled = scaler_y.transform(y_test_2021)
  y_test_2022_scaled = scaler_y.transform(y_test_2022)

  train_dataset = StormDataset(X_train_scaled, y_train_scaled)
  val_dataset = StormDataset(X_val_scaled, y_val_scaled)
  test_dataset = StormDataset(X_test_scaled, y_test_scaled)
  test_dataset_2020 = StormDataset(X_test_2020_scaled, y_test_2020_scaled)
  test_dataset_2021 = StormDataset(X_test_2021_scaled, y_test_2021_scaled)
  test_dataset_2022 = StormDataset(X_test_2022_scaled, y_test_2022_scaled)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
  test_2020_loader = DataLoader(test_dataset_2020, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
  test_2021_loader = DataLoader(test_dataset_2021, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
  test_2022_loader = DataLoader(test_dataset_2022, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
  for model_id in model_ids:
    if model_id == 'rnn_kan_v1_1':
       in_features = (20, 50, 100)
       hidden_features = 100
       output_features = 1
       model = rnn_kan_v1_1.RNN_KAN(in_features, hidden_features, output_features, n_ahead)
    elif model_id == 'rnn_kan_v1_2':
        in_features = (20, 100)
        hidden_features = 100
        output_features = 1
        model = rnn_kan_v1_2.RNN_KAN(in_features, hidden_features, output_features, n_ahead)
       
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model, total_time, train_losses, val_losses = train_val(model, criterion, optimizer, train_loader, val_loader, device, batch_size, epochs)

    mae_test = test(model, test_loader, scaler_y, device)
    mae_test_2020 = test(model, test_2020_loader, scaler_y, device)
    mae_test_2021 = test(model, test_2021_loader, scaler_y, device)
    mae_test_2022 = test(model, test_2022_loader, scaler_y, device)

    results[n_ahead][model_id] = {
        "mae": [mae_test, mae_test_2020, mae_test_2021, mae_test_2022],
        "time": [total_time]
    }

    print(f"{model_id} - {n_ahead * 6}h done! Total time: {total_time:.2f}s")
    print(f"MAE: {mae_test:.6f} | 2020: {mae_test_2020:.6f} | 2021: {mae_test_2021:.6f} | 2022: {mae_test_2022:.6f}")

with open(f"result/performances/exp_storm_torch-{data_id}.txt", "w") as f:
    print("TRAINING SUMMARY")
    f.write("TRAINING SUMMARY\n")

    for n_ahead in n_aheads:
        for model_id in model_ids:
            time_taken = results[n_ahead][model_id]['time'][0]
            mae = results[n_ahead][model_id]['mae'][0]
            mae_2020 = results[n_ahead][model_id]['mae'][1]
            mae_2021 = results[n_ahead][model_id]['mae'][2]
            mae_2022 = results[n_ahead][model_id]['mae'][3]

            line1 = f"{model_id} - {n_ahead * 6}h Prediction - Time: {time_taken:.2f}s"
            line2 = f"{model_id} - {n_ahead * 6}h Prediction - MAE: {mae:.6f}"
            line3 = f"{model_id} - {n_ahead * 6}h Prediction - MAE 2020: {mae_2020:.6f}"
            line4 = f"{model_id} - {n_ahead * 6}h Prediction - MAE 2021: {mae_2021:.6f}"
            line5 = f"{model_id} - {n_ahead * 6}h Prediction - MAE 2022: {mae_2022:.6f}"

            # in ra màn hình
            print(line1); print(line2); print(line3); print(line4); print(line5)

            # ghi vào file
            f.write(line1 + "\n")
            f.write(line2 + "\n")
            f.write(line3 + "\n")
            f.write(line4 + "\n")
            f.write(line5 + "\n")
            f.write("\n")