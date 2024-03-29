from src.db.postgres import Postgres


postgres=Postgres()

# On récupère le nom des différentes tables de la base de donnée, ainsi que le df pour le ticker choisi (nom, ou position dans la liste obtenue)
# table_names=postgres.get_all_table_names()
# ici pour l'exemple on choisi que deux ticker ticker
table_names = ['BTCUSDT', 'ETHEUR']
# Initialiser le dictionnaire de résultats
results = {}
for table in table_names:
    results[table] = []

# Parcourir chaque table et tester chaque combinaison de SMA
for table in table_names:
    df = postgres.get_table_as_dataframe(table)

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
for table in table_names:
    best_result = max(results[table], key=lambda x: x['wallet'])
    print(f"Table {table}: SMA1={best_result['sma1']}, SMA2={best_result['sma2']}, "
          f"Wallet={round(best_result['wallet'], 2)}")


