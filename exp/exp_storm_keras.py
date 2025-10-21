import numpy as np
from collections import defaultdict
import os
from sklearn.metrics import r2_score, mean_absolute_error

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from tkan import TKAN
import time
import gc
import random
import wandb

from utils.data_utils import prepare_data, load_and_process_data, dataset_loader
from utils.model_utils import evaluate_loader
from model.tkan import StormPredictorModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

data_path = 'data/cma-era5-data'
raw_train_data, raw_val_data, raw_test_data = load_and_process_data(data_path=data_path, train_years=list(range(1980, 2017)), val_years=[2017, 2018, 2019], test_years=[2020, 2021, 2022])

# config
batch_size = 128
n_aheads = list(range(1, 13))
sequence_length = 5
lr = 1e-3
model_ids = ["TKAN" ,"LSTM", "GRU"]

results = defaultdict(dict)

# Dùng tất cả GPU
strategy = tf.distribute.MirroredStrategy()
print("GPUs in use:", strategy.num_replicas_in_sync)

num_epochs = 10
# log architecture model
model_ids = [
    "TKAN",
    "GRU",
    "LSTM"
 ]
data_id = 'raw_cma_era5-v0'



for model_id in model_ids:
  model = StormPredictorModel(model_id=model_id, n_ahead=12)
  model.compile(optimizer='adam', loss='mse')
  model.build(input_shape=(batch_size, sequence_length, 12))
  print(model.ts_seq.summary())

for n_ahead in n_aheads:
  X_train, y_train, metadata_train = prepare_data(raw_train_data, sequence_length = sequence_length, n_ahead = n_ahead)
  X_val, y_val, metadata_val = prepare_data(raw_val_data, sequence_length = sequence_length, n_ahead = n_ahead)
  X_test, y_test, metadata_test = prepare_data(raw_test_data, sequence_length = sequence_length, n_ahead = n_ahead)

  print(f"X_shape: {X_train.shape}")
  print(f"y_shape: {y_train.shape}")

  raw_test_data_2020 = {k: v for k, v in raw_test_data.items() if int(k[-4:]) == 2020}
  raw_test_data_2021 = {k: v for k, v in raw_test_data.items() if int(k[-4:]) == 2021}
  raw_test_data_2022 = {k: v for k, v in raw_test_data.items() if int(k[-4:]) == 2022}

  X_test_2020, y_test_2020, metadata_test_2020 = prepare_data(raw_test_data_2020, sequence_length = sequence_length, n_ahead = n_ahead)
  X_test_2021, y_test_2021, metadata_test_2021 = prepare_data(raw_test_data_2021, sequence_length = sequence_length, n_ahead = n_ahead)
  X_test_2022, y_test_2022, metadata_test_2022 = prepare_data(raw_test_data_2022, sequence_length = sequence_length, n_ahead = n_ahead)


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

  train_loader = dataset_loader(X_train_scaled, y_train_scaled, batch_size=batch_size, shuffle=True)
  val_loader   = dataset_loader(X_val_scaled,   y_val_scaled,   batch_size=batch_size, shuffle=False)
  test_loader  = dataset_loader(X_test_scaled,  y_test_scaled,  batch_size=batch_size, shuffle=False)
  test_loader_2020 = dataset_loader(X_test_2020_scaled, y_test_2020_scaled, batch_size=batch_size, shuffle=False)
  test_loader_2021 = dataset_loader(X_test_2021_scaled, y_test_2021_scaled, batch_size=batch_size, shuffle=False)
  test_loader_2022 = dataset_loader(X_test_2022_scaled, y_test_2022_scaled, batch_size=batch_size, shuffle=False)

  for model_id in model_ids:
    print(f"\n=============== Training {model_id} - {n_ahead * 6}h Prediction ===============")
    with strategy.scope():
      model = StormPredictorModel(model_id=model_id, n_ahead=n_ahead)
      optimizer = keras.optimizers.Adam(learning_rate=lr)
      model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())
      model.build(input_shape=(None, sequence_length, 20))

    total_start_time = time.time()

    history = model.fit(train_loader,
                        validation_data=val_loader,
                        epochs=num_epochs,
                        verbose=1)

    total_time = time.time() - total_start_time

    train_losses = history.history['loss']
    val_losses   = history.history['val_loss']

    print(f"\nEvaluating {model_id} on test set...")
    mae_test      = evaluate_loader(model, test_loader, scaler_y)
    mae_test_2020 = evaluate_loader(model, test_loader_2020, scaler_y)
    mae_test_2021 = evaluate_loader(model, test_loader_2021, scaler_y)
    mae_test_2022 = evaluate_loader(model, test_loader_2022, scaler_y)

    results[n_ahead][model_id] = {
        "mae": [mae_test, mae_test_2020, mae_test_2021, mae_test_2022],
        "time": [total_time]
    }


    print(f"{model_id} - {n_ahead * 6}h done! Total time: {total_time:.2f}s")
    print(f"MAE: {mae_test:.6f} | 2020: {mae_test_2020:.6f} | 2021: {mae_test_2021:.6f} | 2022: {mae_test_2022:.6f}")


with open(f"result/performances/exp_storm_keras-{data_id}.txt", "w") as f:
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