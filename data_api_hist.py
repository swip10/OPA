from binance.client import Client
import json
import pandas as pd

api_key = 'eccWMDlc1dNmfGVHz2As8XjibvMrU1Tvm0n8aA5J7oKeF0z3NkwNTWZtHcMHEZv9'
api_secret = 'Sycm1eu55L18GDyBxnDjTnoqbQX0lZdZPe222a6AuX2EThW3VVOhfGtzMqzo8GI9'

# Initialise le client Binance
client = Client(api_key, api_secret)

# Récupère tout les 'tickers' de l'API
tickers = client.get_all_tickers()

# Initialise un dictionnaire pour stocker les données de chaque ticker
ticker_data = {}

# Récupère les données de l'API sur chacun des tickers
for ticker in tickers[0:5]:
    klines = client.get_historical_klines(ticker['symbol'], Client.KLINE_INTERVAL_1DAY, "10 days ago")

    # Transforme les données en DataFrame
    data = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])

    # Change le type de la colonne timestamp en datetime
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms").dt.strftime('%Y-%m-%d %H:%M:%S')

    # Stocke la DataFrame correspondante dans le dictionnaire
    ticker_data[ticker['symbol']] = data.to_dict(orient='records')
    print("OK pour le ticker " + ticker['symbol'])

# Exporte les données au format JSON
with open('ticker_data_hist.json', 'w') as f:
    json.dump(ticker_data, f)