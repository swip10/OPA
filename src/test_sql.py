import sqlite3
import json

# Ouverture du fichier JSON
with open('ticker_data_hist.json') as f:
    data = json.load(f)

# Récupération des tickers présent dans le JSON
keys = data.keys()

# Connexion à la base de données
conn = sqlite3.connect('BDD_hist.sqlite')

c = conn.cursor()

# Création de la boucle pour créer et alimenter les tables
for key in keys:
    c.execute(f'''CREATE TABLE IF NOT EXISTS {key}
                (symbol TEXT,
                 timestamp TEXT, 
                 open FLOAT, 
                 high FLOAT, 
                 low FLOAT, 
                 close FLOAT, 
                 volume FLOAT, 
                 close_time TEXT, 
                 quote_asset_volume FLOAT, 
                 number_of_trades INTEGER, 
                 taker_buy_base_asset_volume FLOAT, 
                 taker_buy_quote_asset_volume FLOAT, 
                 PRIMARY KEY (timestamp, symbol))''')

    # Insertion des données dans la table

    for d in data[key]:
        symbol = key
        timestamp = d['timestamp']
        open_price = d['open']
        high = d['high']
        low = d['low']
        close = d['close']
        volume = d['volume']
        close_time = d['close_time']
        quote_asset_volume = d['quote_asset_volume']
        number_of_trades = d['number_of_trades']
        taker_buy_base_asset_volume = d['taker_buy_base_asset_volume']
        taker_buy_quote_asset_volume = d['taker_buy_quote_asset_volume']

        c.execute(f"INSERT INTO {key} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, timestamp, open_price, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume))


# Validation des changements et fermeture de la connexion à la base de données
conn.commit()
conn.close()

