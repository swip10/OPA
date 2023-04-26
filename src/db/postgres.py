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

        self.cursor.execute(f"SELECT * FROM {ticker}")
        table_data = self.cursor.fetchall()
        table_columns = [desc[0] for desc in self.cursor.description]
        return pd.DataFrame(table_data, columns=table_columns)


