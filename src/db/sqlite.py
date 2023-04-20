import sqlite3
import pathlib
from src.db.sql import SQL


class SQLiteOPA(SQL):
    FILE = (pathlib.Path(__file__).parents[2].resolve()) / "data" / "BDD_hist.sqlite"

    def __init__(self, check_same_thread: bool = True):
        connection = sqlite3.connect(SQLiteOPA.FILE, check_same_thread=check_same_thread)
        super().__init__(connection)
