from binance.client import Client
from config import config
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import plotly.graph_objects as go


PLOT = True
TESTING = True
NUMBER_OF_PREDICTIONS = 5

ticker = "ETHBTC"  # model has been trained with this ticker

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
json_data = {"price": list(data["close"][:59]), "volume": list(data["volume"][:59]), "currency": ticker}

url = "http://127.0.0.1:8000/prediction"

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

params = {
    'next_hours': str(NUMBER_OF_PREDICTIONS*8),
}

r = requests.post(url, params=params, headers=headers, json=json_data)
response = r.json()

if PLOT:
    x_true = np.arange(len(data["close"]))
    y_pred = response["prices"]
    x_pred = np.arange(59, 59 + NUMBER_OF_PREDICTIONS)
    fig = go.Figure(
        go.Scatter(x=x_true, y=data["close"], mode='lines+markers')
    )
    fig.add_trace(
        go.Scatter(x=x_pred, y=y_pred, mode='markers')
    )
    fig.show()
