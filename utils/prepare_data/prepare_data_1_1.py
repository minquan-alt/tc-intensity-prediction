import numpy as np
import gc

'''
cma_features + era5_features --> X (batch, seq_x, n_features).
    cma_features: 4 features (lat, lon, v_max, p_min)
    era5_features:
        single: mỗi features lấy center_grid làm 1 features. có 4 single features => có 4 features
        multi: mỗi features lấy center_grid và 1 depth làm 1 features, có 5 multi features (chỉ lấy 3 features) và 4 depth => 12 features
-> X có 20 features (batch, seq_x, 20)
'''

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