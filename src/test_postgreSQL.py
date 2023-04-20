import json
import psycopg2
import config
from tqdm import tqdm


# Ouverture du fichier JSON:
with open('ticker_data_hist_new.json', 'r') as f:
    ticker_data = json.load(f)

# Connexion à la base de données PostgreSQL OPA_data_hist:
conn = psycopg2.connect(
    port=5432,
    host='localhost',
    database='OPA_data_hist',
    user='postgres2',
    password='ProjetOPA2023$'

)

# Création des tables pour chaque ticker:
for ticker, data in ticker_data.items():
    
    #on modifie les noms de tables pour les ticker commençant par 1INCH car une table ne peut pas commencer par un chiffre

    table_name = ticker if not ticker.startswith("1INCH") else "INCH" + ticker[5:]

    cur = conn.cursor()

    cur.execute(f"CREATE TABLE {table_name} (timestamp TIMESTAMP, open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT, close_time TIMESTAMP, quote_asset_volume FLOAT, number_of_trades INTEGER, taker_buy_base_asset_volume FLOAT, taker_buy_quote_asset_volume FLOAT)")

    # Insertion des données dans la table correspondante:
    for row in tqdm(data, desc=f"Insertion des données dans la table {table_name}"):
        cur.execute(f"INSERT INTO {table_name} (timestamp, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (row["timestamp"], row["open"], row["high"], row["low"], row["close"], row["volume"], row["close_time"], row["quote_asset_volume"], row["number_of_trades"], row["taker_buy_base_asset_volume"], row["taker_buy_quote_asset_volume"]))

    conn.commit()
    cur.close()

# Fermer la connexion
conn.close()