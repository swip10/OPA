import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
from time_serie_base_line_model import TimeSerieBaseLineModel
from config import config


# good scenario 4 to test - pass currency type as input
# https://stackoverflow.com/questions/65345953/adding-exogenous-variables-to-my-univariate-lstm-model


with open(config.CHEMIN_JSON_LOCAL, "r") as json_file: 
    hist_data = json.load(json_file)


list_df = []
for key in hist_data:
    sub_df = pd.DataFrame(hist_data[key])
    sub_df["symbol"] = key
    list_df.append(sub_df)
df = pd.concat(list_df, ignore_index=True)

df = df[["close", "symbol", "volume"]]
df = df.astype({'close': 'float', 'volume': 'float'})

# filter on only one currency during dev
df = df.query("symbol == 'BTCEUR'")

# scaler should be only train on test set
scaler = MinMaxScaler()
close_price = df.close.values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_price)
scaler_volume = MinMaxScaler()
scaled_volume = scaler_volume.fit_transform(df.volume.values.reshape(-1, 1))

sequence_len = 60

scaled_data = np.concatenate((scaled_close, scaled_volume), axis=1)


def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])


def get_train_test_sets(data, seq_len, train_frac):
    sequences = split_into_sequences(data, seq_len)
    n_train = int(sequences.shape[0] * train_frac)
    x_train = sequences[:n_train, :-1, :]
    y_train = sequences[:n_train, -1, :]
    x_test = sequences[n_train:, :-1, :]
    y_test = sequences[n_train:, -1, :]
    return x_train, y_train, x_test, y_test


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=False)
x_train, y_train, x_test, y_test = get_train_test_sets(scaled_data, sequence_len, train_frac=0.9)

# y_train = np.expand_dims(y_train[:, 0], axis=-1)
# y_test = np.expand_dims(y_test[:, 0], axis=-1)

batch_size = 124
model = TimeSerieBaseLineModel(
    seq_len=sequence_len,
    input_shape=x_train.shape[-1],
    output_shape=y_train.shape[-1],
    dropout=0.05
)

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
)

model.build((None, sequence_len, x_train.shape[-1]))
# `input_shape` is the shape of the input data
# e.g. input_shape = (None, 32, 32, 3)
model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=batch_size,
    shuffle=False,
    validation_data=(x_test, y_test)
)

plt.plot(history.history['loss'], label="train")
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)

# extract only the first value prediction i.e. the predicted price
y_train = np.expand_dims(y_train[:, 0], axis=-1)
y_test = np.expand_dims(y_test[:, 0], axis=-1)
y_pred = np.expand_dims(y_pred[:, 0], axis=-1)
y_pred_train = np.expand_dims(y_pred_train[:, 0], axis=-1)

# invert the scaler to get the absolute price data
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred)
y_pred_train_orig = scaler.inverse_transform(y_pred_train)

plt.plot(np.arange(0, len(y_train)), scaler.inverse_transform(y_train), color='brown', label='Historical Price')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test_orig)), y_test_orig, color='orange', label='Actual Price')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_pred_orig)), y_pred_orig, color='green', label='Predicted Price')
plt.plot(np.arange(0, len(y_train)), y_pred_train_orig, color='blue', label='Predicted Price train')

plt.title('ETHBTC 8hours Prices')
plt.xlabel('8hours space')
plt.ylabel('Price ($)')
plt.legend()
plt.show()
