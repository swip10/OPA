import matplotlib.pyplot as plt
from src.db.postgres import Postgres


# Établir une connexion à la base de données PostgreSQLs
postgres_client = Postgres()

#Récupère tous les noms de table de la BDD et les stocke dans une variable table_names
table_names = postgres_client.get_all_table_names()
#print(table_names)

#Création d'un dictionnaire pour stocker les différents dataframes crées:

dfs={}

# Exécuter une requête SQL pour récupérer les données de la table souhaitée

for table_name in table_names:

    # Stocker les résultats de la requête dans le dictionnaire  (nom de la table --> dataframe pandas nommé d'après la table requétée)
    dfs[table_name[0]] = postgres_client.get_data_frame_from_ticker(table_name[0])


# Fermer la connexion à la base de données
postgres_client.close()

#Imprime un aperçu des df ainsi obtenus

#for table_name, df in dfs.items():
    #print(f"Table name: {table_name}")
    #print(df.head())

ticker = "btceur"

# Récupérer le dataframe souhaité (par exemple, celui de la première table)
df = dfs[f"{ticker}"]

# Tracer un graphique avec les données de la colonne "close"
fig = plt.plot(df["close"])
plt.title(f"Close price for {ticker} over the last 1k days")
plt.xlabel("Days")
plt.ylabel("Price")

# Afficher le graphique
plt.show()
