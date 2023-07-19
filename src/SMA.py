from src.db.postgres import Postgres
from tqdm import tqdm

postgres=Postgres()

# On récupère le nom des différentes tables de la base de donnée, ainsi que le df pour le ticker choisi (nom, ou position dans la liste obtenue)
table_names=postgres.get_all_table_names()
table_names = [name[0] for name in table_names]
# ici pour l'exemple on choisi que deux ou trois ticker 
# table_names = ['btcusdt', 'subbtc','ltceur']

# Initialisation d'une liste pour les tickers pour lesquels on ne peux pas appliquer la stratégie
ticker_impossible = []

# Initialiser le dictionnaire des wallets pour chaque table
wallets = {}
for table in table_names:
    wallets[table] = 1000

# Parcourir chaque table et calculer les moyennes mobiles
for table in tqdm(table_names):
    df = postgres.get_table_as_dataframe(table)
    # Skip si pas de colonne 'close' dans les colonnes (autres tables que les tickers)
    if 'close' not in df.columns:
        continue
    # Skip si le dataframe est vide
    if df.empty or len(df) < 200:
        ticker_impossible.append(table)
        wallets.pop(table, None)
        continue


    df['sma20'] = df['close'].rolling(window=20).mean().iloc[49:]
    df['sma50'] = df['close'].rolling(window=50).mean().iloc[49:]

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
#for wallet in wallets:
#    if wallet != max_wallet:
#        print(f"Le wallet {wallet} a {round(wallets[wallet], 2)}$ à la fin.")

# Affichage des erreurs
print(f"Nombre ticker(s) pour lesquel la stratégie n'a pas pu être appliquée: {len(ticker_impossible)}/ {len(table_names)} \n"
      f"Exemple de ticker(s) pour lesquel la stratégie n'a pas pu être appliquée: {ticker_impossible[:5]}")
