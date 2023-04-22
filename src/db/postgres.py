import psycopg2

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

