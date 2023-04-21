import sqlite3
import matplotlib.pyplot as plt
import json
import sql_client
import pandas as pd

# Connexion à la base de données
conn = sql_client.get_db_client()

# Créer un curseur pour exécuter des requêtes SQL
c = conn.cursor()

# Récupérer les données de clôture pour un ticker spécifique
symbol = 'BTCEUR'
c.execute(f"SELECT timestamp, close FROM {symbol}")
data = c.fetchall()

# Charger les données dans un DataFrame
df = pd.read_sql_query("SELECT * FROM BTCEUR", conn, index_col='timestamp')

# Calculer les moyennes mobiles sur les colonnes 'close'
sma_20 = df['close'].rolling(window=20).mean()
sma_50 = df['close'].rolling(window=50).mean()

# Séparer les données en listes de temps et de prix
timestamps = [row[0] for row in data]
prices = [row[1] for row in data]

# Créer le graphique linéaire
plt.plot(timestamps, prices)
plt.xlabel('Temps')
plt.ylabel('Prix de clôture')
plt.title(f'Histoire des prix de clôture pour {symbol}')
plt.plot(df['close'], label='BTCUSDT')
plt.plot(sma_20, label='SMA 20')
plt.plot(sma_50, label='SMA 50')
plt.legend()
plt.show()


# Fermer la connexion à la base de données
conn.close()