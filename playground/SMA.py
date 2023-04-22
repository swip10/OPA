from src.db.sqlite import SQLiteOPA


# Se connecter à la base de données
sqlite_client = SQLiteOPA()

# Récupérer les noms des tables
table_names = sqlite_client.get_all_table_names()
print(table_names)

# Initialiser le dictionnaire des wallets pour chaque table
wallets = {}
for table in table_names:
    wallets[table] = 1000

# Parcourir chaque table et calculer les moyennes mobiles
for table in table_names:
    df = sqlite_client.get_data_frame_from_ticker(table)
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
print(f"Le wallet avec le plus d'argent est {max_wallet} avec {round(wallets[max_wallet], 2)}$ à la fin.")

# Afficher les autres wallets
for wallet in wallets:
    if wallet != max_wallet:
        print(f"Le wallet {wallet} a {round(wallets[wallet], 2)}$ à la fin.")