from binance.client import Client
import json
from tqdm import tqdm
import pandas as pd
import config


# Initialise le client Binance
client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
client.ping()

# Récupère tout les 'tickers' de l'API
tickers = client.get_all_tickers()

# Initialise un dictionnaire pour stocker les données de chaque ticker
ticker_data = {}
loading_tickers = tqdm(tickers[0:10])
# Récupère les données de l'API sur chacun des tickers
for ticker in loading_tickers:
    loading_tickers.set_description(f"loading symbol {ticker['symbol']}")
    klines = client.get_historical_klines(ticker['symbol'], Client.KLINE_INTERVAL_8HOUR, "1000 days ago")

    # Transforme les données en DataFrame
    data = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    data.drop(columns="ignore", inplace=True)

    # Change le type de la colonne timestamp en datetime
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms").dt.strftime('%Y-%m-%d %H:%M:%S')

    # Stocke la DataFrame correspondante dans le dictionnaire
    ticker_data[ticker['symbol']] = data.to_dict(orient='records')

# Exporte les données au format JSON
with open('ticker_data_hist.json', 'w') as f:
    json.dump(ticker_data, f)
