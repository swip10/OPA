import sqlite3
import json
import sql_client


# Ouverture du fichier JSON
with open('../ticker_data_hist.json') as f:
    data = json.load(f)

# Récupération des tickers présent dans le JSON
keys = data.keys()

# Connexion à la base de données
conn = sql_client.get_db_client()
c = conn.cursor()

# Création de la boucle pour créer et alimenter les tables
for key in keys:
    sql_client.create_table_database(c, key)

    # Insertion des données dans la table
    for d in data[key]:
        print(key)
        sql_client.add_line_to_database(d, key, conn, close=False)

# Validation des changements et fermeture de la connexion à la base de données
conn.close()
