import sqlite3
import pandas as pd

# Se connecter à la base de données
conn = sqlite3.connect('BDD_hist.sqlite')

# Vérifier la connexion
if conn:
    print("La connexion à la base de données est établie.")
else:
    print("La connexion à la base de données a échoué.")

# Récupérer les noms des tables
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in c.fetchall()]
print(tables)

# Initialiser le dictionnaire des wallets pour chaque table
wallets = {}
for table in tables:
    wallets[table] = 1000

# Parcourir chaque table et calculer les moyennes mobiles
for table in tables:
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    df.set_index('timestamp', inplace=True)
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()

    # Analyser les croisements des moyennes mobiles et acheter ou vendre
    in_position = False
    for i in range(len(df)):
        if df['sma20'][i] > df['sma50'][i] and not in_position:
            buy_price = df['close'][i]
            in_position = True
        elif df['sma20'][i] < df['sma50'][i] and in_position:
            sell_price = df['close'][i]
            profit = (sell_price - buy_price) * (wallets[table] / buy_price)
            wallets[table] += profit
            in_position = False

# Trouver le wallet avec le plus d'argent à la fin
max_wallet = max(wallets, key=wallets.get)
print(f"Le wallet avec le plus d'argent est {max_wallet} avec {wallets[max_wallet]}$ à la fin.")

# Afficher les autres wallets
for wallet in wallets:
    if wallet != max_wallet:
        print(f"Le wallet {wallet} a {wallets[wallet]}$ à la fin.")