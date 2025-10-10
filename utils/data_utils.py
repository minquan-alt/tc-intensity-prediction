import os
import pickle
import numpy as np
import torch
import gc
from torch.utils.data import Dataset, DataLoader

def get_raw_data(raw_data_dict):
  raw_data = {}
  for dict_ in raw_data_dict.values():
    raw_data.update(dict_)
  return raw_data
def load_and_process_data(data_path, train_years, val_years, test_years):
  raw_train_data_dict = {}
  raw_val_data_dict = {}
  raw_test_data_dict = {}
  for filename in os.listdir(data_path):
    if filename.endswith('.pkl'):
      year = int(filename[11:15])
      if year not in train_years and year not in val_years and year not in test_years:
        continue
      with open(os.path.join(data_path, filename), 'rb') as f:
        data = pickle.load(f)
        if year in train_years:
          raw_train_data_dict[filename] = data
        elif year in val_years:
          raw_val_data_dict[filename] = data
        elif year in test_years:
          raw_test_data_dict[filename] = data
        else:
          raise ValueError(f"Invalid year: {year}")

  raw_train_data_dict = dict(sorted(raw_train_data_dict.items()))
  raw_val_data_dict = dict(sorted(raw_val_data_dict.items()))
  raw_test_data_dict = dict(sorted(raw_test_data_dict.items()))

  raw_train_data = get_raw_data(raw_train_data_dict)
  raw_val_data = get_raw_data(raw_val_data_dict)
  raw_test_data = get_raw_data(raw_test_data_dict)
  return raw_train_data, raw_val_data, raw_test_data

def prepare_data(storm_data, sequence_length, n_ahead, dtype=np.float32):
  total_sequence = 0
  center_grid = 15
  for sid, storm_records in storm_data.items():
    if len(storm_records) < sequence_length + n_ahead:
      continue
    total_sequence += len(storm_records) - sequence_length - n_ahead + 1

  first_key = next(iter(storm_data.keys()))

  cma_len = len(storm_data[first_key][0]['targets'])
  era5_single_len = storm_data[first_key][0]['features']['single'].shape[0]
  era5_multi_len = storm_data[first_key][0]['features']['multi'][1:4].shape[0] * storm_data[first_key][0]['features']['multi'].shape[1]
  features_len = cma_len + era5_single_len + era5_multi_len
  input_shape = (total_sequence, sequence_length, features_len)
  output_shape = (total_sequence, n_ahead)

  X_sequences = np.empty(input_shape, dtype=dtype)
  y_sequences = np.empty(output_shape, dtype=dtype)
  sequence_metadata = [None] * total_sequence

  valid_storms = 0
  idx = 0

  for sid, storm_records in storm_data.items():
    if len(storm_records) < sequence_length + n_ahead:
      continue

    valid_storms += 1
    L = len(storm_records) - sequence_length - n_ahead + 1

    for i in range(L):
      for j in range(sequence_length):
        target = storm_records[i + j]['targets']
        cma_features = dtype([target['center_lat'],target['center_lon'],target['vmax'],target['pmin']])

        era5_features = []
        single_era5_features = storm_records[i + j]['features']['single']
        multi_era5_features = storm_records[i + j]['features']['multi'][1:4, :, :, :]

        for m in range(single_era5_features.shape[0]):
          era5_features.append(single_era5_features[m, center_grid, center_grid])
        for m in range(multi_era5_features.shape[0]):
          for n in range(multi_era5_features.shape[1]):
            era5_features.append(multi_era5_features[m, n, center_grid, center_grid])

        era5_features = dtype(era5_features)

        X_sequences[idx, j, :4] = cma_features
        X_sequences[idx, j, 4:] = era5_features

      for j in range(n_ahead):
        target = storm_records[i + sequence_length + j]['targets']
        y_sequences[idx, j] = dtype(target['vmax'])

      sequence_metadata[idx] = {
        'storm_id': sid,
        'input_times': [storm_records[i + j]['time'] for j in range(sequence_length)],
        'target_time': [storm_records[i + sequence_length + j]['time'] for j in range(n_ahead)]
      }

      idx += 1
  if idx < total_sequence:
    X_sequences = X_sequences[:idx]
    y_sequences = y_sequences[:idx]
    sequence_metadata = sequence_metadata[:idx]

  metadata = {
    'n_sequences': idx,
    'sequence_length': sequence_length,
    'n_storms': valid_storms,
    'sequence_metadata': sequence_metadata,
  }

  gc.collect()
  return (X_sequences, y_sequences, metadata)

class StormDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]