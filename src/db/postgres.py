import psycopg2
import pandas as pd
from config import config
from src.db.sql import SQL


class Postgres(SQL):

    def __init__(self):
        connection = psycopg2.connect(
            port=config.port,
            host=config.host,
            database=config.database,
            user=config.db_user,
            password=config.db_password
        )
        super().__init__(connection)

    def get_all_table_names(self) -> list[str]:
        self.cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'"
        )
        table_names = self.cursor.fetchall()
        return table_names
    
    #Fonction permettant de récupérer un df à partir d'une table d'un ticker spécifié:
    
    def get_table_as_dataframe(self, ticker: str) -> pd.DataFrame:
        try:
            self.cursor.execute(f"SELECT * FROM {ticker}")
            table_data = self.cursor.fetchall()
            table_columns = [desc[0] for desc in self.cursor.description]
            return pd.DataFrame(table_data, columns=table_columns)
        except psycopg2.errors.UndefinedTable:
            return pd.DataFrame()  # Retourne un dataframe vide
        except Exception as e:
            return pd.DataFrame()  # Retourne un dataframe vide
        
    def create_volatility_table(self):
        create_table_query = """
            CREATE TABLE IF NOT EXISTS VOLATILITE(
                Coin VARCHAR(50),
                Rang VARCHAR(10),
                Volatilite_1m VARCHAR(10),
                prix VARCHAR(20),
                variation_pourcentage_24h VARCHAR(10),
                capitalisation_boursiere VARCHAR(20),
                volume_en_usd_24h VARCHAR(20),
                actifs_circulants VARCHAR(20),
                categorie VARCHAR(200)
            );
        """
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            print("Table created successfully in PostgreSQL ")
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error while creating PostgreSQL table", error)



    def insert_df_into_table(self, df: pd.DataFrame, table_name: str) -> None:
        # Get the column names from the SQL table
        self.cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
        colnames = [desc[0] for desc in self.cursor.description]

        # Filter the DataFrame to only include columns that exist in the SQL table
        df = df[colnames]

        # Convert the DataFrame to a list of tuples
        tuples = [tuple(x) for x in df.to_numpy()]

        # Create a string of the column names
        cols = ','.join(list(df.columns))

        # Create a string of substitution characters
        vals = ','.join(['%s' for _ in range(len(df.columns))])

        # Build the SQL query
        query = "INSERT INTO %s(%s) VALUES(%s)" % (table_name, cols, vals)

        # Execute the SQL query
        try:
            self.cursor.executemany(query, tuples)
            self.connection.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.connection.rollback()



    def drop_table(self, table_name: str) -> None:
        """Supprimer une table dans la base de données."""
        try:
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.connection.commit()
            print(f"Table {table_name} dropped successfully.")
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.connection.rollback()