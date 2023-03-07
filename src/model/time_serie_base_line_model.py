from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model

# extracted from https://medium.com/codex/time-series-prediction-using-lstm-in-python-19b1187f580f


class TimeSerieBaseLineModel(Model):

    def __init__(self, seq_len, input_shape, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.window_size = seq_len - 1
        self.lstm1 = LSTM(self.window_size, return_sequences=True, input_shape=(self.window_size, input_shape))
        self.dropout1 = Dropout(rate=dropout)
        self.lstm2 = Bidirectional(LSTM((self.window_size * 2), return_sequences=True))
        self.dropout2 = Dropout(rate=dropout)
        self.lstm3 = Bidirectional(LSTM((self.window_size * 2), return_sequences=False))
        self.dense1 = Dense(units=1, activation="linear")

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dropout2(x)
        x = self.lstm3(x)
        return self.dense1(x)
