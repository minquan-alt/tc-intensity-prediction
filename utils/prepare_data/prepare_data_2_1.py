import numpy as np

'''
cma_features + era5_features --> X_multi(batch, seq_x, n_multi_features, depths, width, height), X_single(batch, seq_x, n_single_features).
    cma_features: 4 single features -> X_single
    era5_features:
        single: có 4 2D features => 4 features (width, height)
        multi: có 5 3D features => 5 features (depth, width, height)
        concat single+multi theo depth -> 9 multi features -> X_multi
    -> Có 13 features nhưng các multi features có chiều không gian cao hơn (6D)
'''

def prepare_data(storm_data, sequence_length, n_ahead, dtype=np.float32):
    X_sequences_multi = []
    X_sequences_single = []
    sequence_metadata = []

    valid_storms = 0
    total_sequence = 0
    for sid, storm_records in storm_data.items():
        if len(storm_records) < sequence_length + n_ahead:
            continue
        total_sequence += len(storm_records) - sequence_length - n_ahead + 1

    y_sequences = np.empty((total_sequence, n_ahead))
    idx = 0

    for sid, storm_records in storm_data.items():
        if len(storm_records) < sequence_length + 1:  # Cần ít nhất 5 time steps (4 input + 1 target)
            print(f"Bỏ qua storm {sid}: chỉ có {len(storm_records)} time steps (cần ít nhất {sequence_length + 1})")
            continue

        valid_storms += 1
        L = len(storm_records) - sequence_length - n_ahead + 1

        for i in range(L):
            input_sequence = {
                'multi': [],
                'single': []
            }
            for j in range(sequence_length):
                target = storm_records[i + j]['targets']
                cma_features = dtype([target['center_lat'],target['center_lon'],target['vmax'],target['pmin']])
                cma_features = np.array(cma_features)


                single_era5_features = storm_records[i + j]['features']['single']
                multi_era5_features = storm_records[i + j]['features']['multi']
                single_era5_features = single_era5_features[None, :, :, :]
                single_era5_features = np.array(single_era5_features)
                multi_era5_features = np.array(multi_era5_features)
                era5_features = np.concatenate([single_era5_features, multi_era5_features], axis=0)

                input_sequence['single'].append(cma_features)
                input_sequence['multi'].append(era5_features)
            

            # input_sequence['single'].append(generate_single_data(storm_record_targets, first_storm_time))
            for j in range(n_ahead):
                target = storm_records[i + sequence_length + j]['targets']
                y_sequences[idx, j] = dtype(target['vmax'])

            X_sequences_multi.append(input_sequence['multi'])
            X_sequences_single.append(input_sequence['single'])

            sequence_metadata.append({
                'storm_id': sid,
                'input_times': [storm_records[i + j]['time'] for j in range(sequence_length)],
                'target_time': storm_records[i + sequence_length]['time']
            })

            idx += 1

    X_multi = np.array(X_sequences_multi)
    X_single = np.array(X_sequences_single)
    y = np.array(y_sequences)

    metadata = {
        'n_sequences': idx,
        'sequence_length': sequence_length,
        'n_storms': valid_storms,
        'sequence_metadata': sequence_metadata,
    }

    # print(f"\nData preparation completed:")
    # print(f"  Số cơn bão: {valid_storms}")
    # print(f"  Số sequences: {len(X_sequences_multi)}")
    # print(f"  Shape X_multi: {X_multi.shape}")
    # print(f"  Shape X_single: {X_single.shape}")
    # print(f"  Shape y: {y.shape}")

    return X_multi, X_single, y, metadata