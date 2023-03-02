from binance.client import Client
import pandas as pd
import config
from time import time, sleep


TIMEOUT = 500  # timeout in seconds

# Initialise le client Binance
client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
client.ping()

symbols = ["ETHBTC", "LTCBTC", "BNBBTC", "NEOBTC", "QTUMETH"]

# transformer la liste en une string pour la requete
# %5B%22ETHBTC%22,%22LTCBTC%22%5D pour la liste ETHBTC, LTCBTC
string_request = "%5B" + ",".join([f"%22{symbol}%22" for symbol in symbols]) + "%5D"

init_time = time()

ticker_data = []
while time() - init_time < TIMEOUT:
    latest_price = client.get_orderbook_tickers(symbols=string_request)
    timestamp = time()
    data = pd.DataFrame(latest_price, columns=["symbol", "bidPrice", "bidQty", "askPrice", "askQty"])
    data["timestamp"] = timestamp
    ticker_data.append(data)
    sleep(20)


data = pd.concat(ticker_data)

data.to_csv("plots/ticker_latest_prices.csv")
