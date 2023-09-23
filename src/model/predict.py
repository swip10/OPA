from binance.client import Client
from config import config
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from keras.saving.saving_api import load_model
from pathlib import Path
import plotly.graph_objects as go


PLOT = True
TESTING = True
NUMBER_OF_PREDICTIONS = 5

ticker = "ETHBTC"  # model has been trained with this ticker
the_script = Path(__file__)

client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
client.ping()

nb_samples = 8 * 59  # 59 values spaced by 8 jours
nb_samples += NUMBER_OF_PREDICTIONS*8 if TESTING is True else 0  # keep true value to compare in testing mode
klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_8HOUR, f"{nb_samples} hours ago")
data = pd.DataFrame(
    klines,
    columns=[
        "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
)
data = data[["close", "volume"]]
data = data.astype(float)

volume_scaler = joblib.load(the_script.parents[2] / "models" / "001_close_volume" / "scaler_volume.save")
price_scaler = joblib.load(the_script.parents[2] / "models" / "001_close_volume" / "scaler_close_price.save")

close = price_scaler.transform(data.close.values.reshape(-1, 1))
volume = volume_scaler.transform(data.volume.values.reshape(-1, 1))

x = np.array([close, volume]).T
xx = tf.convert_to_tensor(x, np.float32)
x = xx 
model = load_model(the_script.parents[2] / "models" / "001_close_volume" / "keras_next")

predictions = []
for i in range(NUMBER_OF_PREDICTIONS):
    if TESTING:
        x = xx[:, i:i+59, :]
    y = model.predict(x)[0]
    if not TESTING:
        x = tf.concat([x[:, 1:, :], [[[y[0], y[1]]]]], axis=1)
    predictions.append(y)
predictions = np.array(predictions)

if PLOT:
    x_true = np.arange(len(data["close"]))
    y_pred = price_scaler.inverse_transform(predictions.T[0].reshape(-1, 1)).reshape(NUMBER_OF_PREDICTIONS)
    x_pred = np.arange(59, 59 + NUMBER_OF_PREDICTIONS)
    fig = go.Figure(
        go.Scatter(x=x_true, y=data["close"], mode='lines+markers')
    )
    fig.add_trace(
        go.Scatter(x=x_pred, y=y_pred, mode='markers')
    )
    fig.show()
