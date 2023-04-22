import matplotlib.pyplot as plt
from src.db.sqlite import SQLiteOPA


# Se connecter à la base de données
sqlite_client = SQLiteOPA()

# Récupérer les données de clôture pour un ticker spécifique
symbol = 'BTCEUR'

# Charger les données dans un DataFrame
df = sqlite_client.get_data_frame_from_ticker(symbol)

# Calculer les moyennes mobiles sur les colonnes 'close'
sma_20 = df['close'].rolling(window=20).mean()
sma_50 = df['close'].rolling(window=50).mean()

# Séparer les données en listes de temps et de prix
timestamps = df.index
prices = df['close']

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
sqlite_client.close()
