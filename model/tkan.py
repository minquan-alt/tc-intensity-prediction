from tensorflow import keras
from keras import layers
from keras.models import Sequential
from tkan import TKAN

class StormPredictorModel(keras.Model):
  def __init__(self, model_id='TKAN', n_ahead=1):
    super().__init__()

    if model_id == 'TKAN':
      self.ts_seq = Sequential([
          TKAN(100, return_sequences=True),
          TKAN(100, sub_kan_output_dim = 20, sub_kan_input_dim = 20, return_sequences=False),
          layers.Dense(units=n_ahead, activation='linear')
      ], name = model_id)
    elif model_id == 'GRU':
      self.ts_seq = Sequential([
        layers.GRU(100, return_sequences=True),
        layers.GRU(100, return_sequences=False),
        layers.Dense(units=n_ahead, activation='linear')
      ], name = model_id)
    elif model_id == 'LSTM':
      self.ts_seq = Sequential([
        layers.LSTM(100, return_sequences=True),
        layers.LSTM(100, return_sequences=False),
        layers.Dense(units=n_ahead, activation='linear')
      ], name = model_id)
    else:
      raise ValueError

  def build(self, input_shape):
    # input_shape = [(batch, seq, features, channels, H, W), (batch, single_dim)]
    super().build(input_shape)
    self.ts_seq.build(input_shape)

  def call(self, X):
    x = self.ts_seq(X)
    return x