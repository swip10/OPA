from binance.client import Client
import pandas as pd
import config
from time import time, sleep


TIMEOUT = 600  # timeout in seconds

# Initialise le client Binance
client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
client.ping()
listen_key = client.stream_get_listen_key()

symbols = ["ETHBTC", "LTCBTC", "BNBBTC", "NEOBTC", "QTUMETH"]

init_time = time()
ticker_data = {k: [] for k in symbols}
ticker_data["timestamp"] = []
while time() - init_time < TIMEOUT:
    print(time() - init_time)
    timestamp = time()
    ticker_data["timestamp"].append(timestamp)
    for symbol in symbols:
        avg_price = client.get_avg_price(symbol=symbol)  # average price during last 5min
        ticker_data[symbol].append(avg_price["price"])

    client.stream_keepalive(listen_key)
    sleep(60)

pd.DataFrame(ticker_data).to_csv("plots/ticker_avg_price.csv")
