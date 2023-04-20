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
