import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class SequenceBuilder:
    def __init__(self, window_size: int, horizon: int = 1):
        self.window_size = window_size
        self.horizon = horizon

    def build(self, arr: np.ndarray):
        sequences = []
        for i in range(len(arr) - self.window_size - self.horizon + 1):
            X = arr[i:i + self.window_size]
            y = arr[i + self.window_size:i + self.window_size + self.horizon]
            sequences.append((X, y))
        return sequences
    
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y
    
class DataModule:
    def __init__(self, file_path, scaler = StandardScaler(), state=42):
        self.file_path = file_path
        self.scaler = scaler
        self.state = state
        name = file_path.split('/')[-1].split(':')[0].split('.')[0].strip()
        print(name)
        if name == 'raw_ETTh1':
            self.start_date = '2016-07-01 00:00:00'
            self.interval = '1h'
            self.split = (12, 4, 4)
        elif name == 'raw_ETTh2':
            self.start_date = '2016-07-01 00:00:00'
            self.interval = '1h'
            self.split = (12, 4, 4)
        elif name == 'raw_ETTm1':
            self.start_date = '2016-07-01 00:00:00'
            self.interval = '15min'
            self.split = (12, 4, 4)
        elif name == 'raw_electricity':
            self.start_date = None
            self.interval = None
            self.split = (6, 2, 2)
        elif name == 'raw_exchange_rate':
            self.start_date = None
            self.interval = None
            self.split = (6, 2, 2)
        elif name == 'raw_PEMS03':
            self.start_date = None
            self.interval = None
            self.split = (6, 2, 2)
        elif name == 'raw_PEMS04':
            self.start_date = None
            self.interval = None
            self.split = (6, 2, 2)
        elif name == 'raw_PEMS07':
            self.start_date = None
            self.interval = None
            self.split = (6, 2, 2)
        elif name == 'raw_PEMS08':
            self.start_date = None
            self.interval = None
            self.split = (6, 2, 2)
        else:
            raise ValueError(f"Dataset {name} chưa được cấu hình start_date và interval.")

        self.raw_train = None
        self.raw_val = None
        self.raw_test = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_raw(self):
        ext = os.path.splitext(self.file_path)[-1]

        if ext in [".csv", ".txt"]:
            with open(self.file_path, "rb") as f:
                first_line = f.readline().decode("utf-8", errors="ignore")

            if first_line.startswith(','):
                raw = pd.read_csv(self.file_path, index_col=0, low_memory=False)
            else:
                raw = pd.read_csv(self.file_path, low_memory=False)

            if 'date' in raw.columns:
                raw = raw.drop(columns=['date'])

            arr = raw.select_dtypes(include=[np.number]).values

        elif ext in [".npz", ".npy"]:
            arr = np.load(self.file_path)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = list(arr.values())[0]  # lấy mảng đầu tiên

        elif ext in [".pkl", ".pickle"]:
            with open(self.file_path, "rb") as f:
                raw = pickle.load(f)
            if isinstance(raw, pd.DataFrame):
                arr = raw.to_numpy()
            elif isinstance(raw, dict):
                arr = np.array(list(raw.values())).T
            elif isinstance(raw, list):
                arr = np.array(raw)
            elif isinstance(raw, np.ndarray):
                arr = raw
            else:
                raise TypeError("Unsupported data type in pickle file")

        else:
            raise ValueError(f"Unsupported file format: {ext}")

        arr = arr.astype(np.float32)
        return arr
    def scale(self):
        scaler = self.scaler

        orig_train_shape = self.raw_train.shape
        orig_val_shape = self.raw_val.shape
        orig_test_shape = self.raw_test.shape

        def to_2d(arr):
            if arr.ndim == 3:
                N, W, F = arr.shape
                return arr.reshape(N * W, F), True
            elif arr.ndim == 2:
                return arr, False
            else:
                raise ValueError(f"Array dim {arr.ndim} không được hỗ trợ")
        
        train_2d, train_was_3d = to_2d(self.raw_train)
        val_2d, val_was_3d     = to_2d(self.raw_val)
        test_2d, test_was_3d   = to_2d(self.raw_test)

        scaler.fit(train_2d)

        train_s = scaler.transform(train_2d)
        val_s   = scaler.transform(val_2d)
        test_s  = scaler.transform(test_2d)

        train_s = train_s.astype(np.float32)
        val_s   = val_s.astype(np.float32)
        test_s  = test_s.astype(np.float32)

        if train_was_3d:
            N, W, F = orig_train_shape
            self.raw_train = train_s.reshape(N, W, F)
        else:
            self.raw_train = train_s.reshape(orig_train_shape)

        if val_was_3d:
            N, W, F = orig_val_shape
            self.raw_val = val_s.reshape(N, W, F)
        else:
            self.raw_val = val_s.reshape(orig_val_shape)

        if test_was_3d:
            N, W, F = orig_test_shape
            self.raw_test = test_s.reshape(N, W, F)
        else:
            self.raw_test = test_s.reshape(orig_test_shape)

    def inverse_scale(self, data):
        orig_shape = data.shape

        if data.ndim == 3:
            N, W, F = orig_shape
            data_2d = data.reshape(N * W, F)
            inv_2d = self.scaler.inverse_transform(data_2d)
            inv = inv_2d.reshape(N, W, F)
        elif data.ndim == 2:
            inv_2d = self.scaler.inverse_transform(data)
            inv = inv_2d
        else:
            raise ValueError(f"Array dim {orig_shape} không được hỗ trợ")

        return inv.astype(np.float32)

    def setup(self):
        arr = self.load_raw()

        if self.start_date is None:
          raw_train, raw_test = train_test_split(arr, test_size=(self.split[1] + self.split[2]) / (self.split[0] + self.split[1] + self.split[2]), random_state=self.state, shuffle=False)
          raw_val, raw_test = train_test_split(raw_test, test_size=self.split[2] / (self.split[1] + self.split[2]), random_state=self.state)

        else:
            orig_shape = arr.shape
            # if arr.shape[-1] == 1:
            #     arr = arr.squeeze(axis=-1)  # (n, features)

            if arr.ndim == 3:
                arr = arr.reshape(arr.shape[0], -1)  # (n, features*? )

            date = pd.date_range(start=self.start_date, periods=arr.shape[0], freq=self.interval).to_pydatetime().tolist()

            if len(date) != arr.shape[0]:
                raise ValueError("Length of date must match number of rows")

            df_tmp = pd.DataFrame(arr)
            df_tmp['date'] = date

            # gom theo tháng
            df_tmp['year_month'] = df_tmp['date'].dt.to_period('M')
            groups = [g for _, g in df_tmp.groupby('year_month')]
            #   total = len(groups)
            #   ratios = [s / sum(self.split) for s in self.split]
            #   months = [round(total * rate) for rate in ratios]

            #   remainder = total - sum(months)
            #   months[0] += remainder
            
            months = self.split
            train_groups = groups[:months[0]]
            val_groups = groups[months[0]:months[0] + months[1]]
            test_groups = groups[months[0] + months[1]: months[0] + months[1] + months[2]]

            train_df = pd.concat(train_groups, axis=0, ignore_index=True)
            val_df = pd.concat(val_groups, axis=0, ignore_index=True)
            test_df = pd.concat(test_groups, axis=0, ignore_index=True)

            raw_train = train_df.drop(columns=['date','year_month']).values
            raw_val = val_df.drop(columns=['date','year_month']).values
            raw_test = test_df.drop(columns=['date','year_month']).values

            if len(orig_shape) == 3:
                # orig_shape: (n, f, 1) / (n,f,g)
                n = raw_train.shape[0]
                features_shape = orig_shape[1:]  # (f, 1), (f, g)
                raw_train = raw_train.reshape((n,)+features_shape)

                n = raw_val.shape[0]
                raw_val = raw_val.reshape((n,)+features_shape)

                n = raw_test.shape[0]
                raw_test = raw_test.reshape((n,)+features_shape)

        self.raw_train = raw_train
        self.raw_val = raw_val
        self.raw_test = raw_test
    def get_dataloader(self, batch_size, window_size, horizon):
        builder = SequenceBuilder(window_size, horizon)

        sequences_train = builder.build(self.raw_train)
        sequences_val   = builder.build(self.raw_val)
        sequences_test  = builder.build(self.raw_test)

        self.train_dataset = SequenceDataset(sequences_train)
        self.val_dataset   = SequenceDataset(sequences_val)
        self.test_dataset  = SequenceDataset(sequences_test)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader