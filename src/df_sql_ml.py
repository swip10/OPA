import psycopg2
import pandas as pd
import config

# Établir une connexion à la base de données PostgreSQL
conn = psycopg2.connect(
    port=5432,
    host='localhost',
    database='OPA_data_hist',
    user='postgres2',
    password='ProjetOPA2023$'
)

#Récupère tous les noms de table de la BDD et les stocke dans une variable table_names
cur = conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'")
table_names = cur.fetchall()
cur.close()
print(table_names)

#Création d'un dictionnaire pour stocker les différents dataframes crées:

dfs={}

# Exécuter une requête SQL pour récupérer les données de la table souhaitée

for table_name in table_names:

    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name[0]};")


# Stocker les résultats de la requête dans le dictionnaire  (nom de la table --> dataframe pandas nommé d'après la table requétée)

    dfs[table_name[0]] = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    cur.close()

# Fermer la connexion à la base de données
conn.close()

#Imprime un aperçu des df ainsi obtenus

for table_name, df in dfs.items():
    print(f"Table name: {table_name}")
    print(df.head())