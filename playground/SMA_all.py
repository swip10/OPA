import sqlite3
import pandas as pd

# Se connecter à la base de données
conn = sqlite3.connect('BDD_hist.sqlite')

# Récupérer les noms des tables
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in c.fetchall()]

# Initialiser le dictionnaire de résultats
results = {}
for table in tables:
    results[table] = []

# Parcourir chaque table et tester chaque combinaison de SMA
for table in tables:
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    df.set_index('timestamp', inplace=True)

    # Initialiser le dictionnaire des wallets pour chaque combinaison de SMA
    wallets = {}
    for i in range(20, 30, 2):
        for j in range(50, 201, 10):
            if i < j:
                wallets[(i, j)] = 1000

    # Parcourir chaque combinaison de SMA et calculer les résultats
    for i in range(20, 30, 2):
        for j in range(50, 201, 10):
            if i < j:
                df['sma'+str(i)] = df['close'].rolling(window=i).mean()
                df['sma'+str(j)] = df['close'].rolling(window=j).mean() 

                # Analyser les croisements des moyennes mobiles et acheter ou vendre
                in_position = False
                for k in range(len(df)):
                    if df['sma'+str(i)][k] > df['sma'+str(j)][k] and not in_position:
                        buy_price = df['close'][k]
                        in_position = True
                        wallet = wallets[(i, j)]
                    elif df['sma'+str(i)][k] < df['sma'+str(j)][k] and in_position:
                        sell_price = df['close'][k]
                        profit = (sell_price - buy_price) * (wallet / buy_price)
                        wallet += profit
                        wallets[(i, j)] = wallet
                        in_position = False

                # Ajouter les résultats pour cette combinaison de SMA
                results[table].append({
                    'sma1': i,
                    'sma2': j,
                    'wallet': max(wallets.values())
                })

# Afficher les résultats
# Trouver le meilleur résultat pour chaque table
for table in tables:
    best_result = max(results[table], key=lambda x: x['wallet'])
    print(f"Table {table}: SMA1={best_result['sma1']}, SMA2={best_result['sma2']}, Wallet={best_result['wallet']}")

