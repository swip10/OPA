import pandas as pd
from src.db.clientdb import DBClient


class SQL(DBClient):

    def __init__(self, connection):
        super().__init__()
        self.connection = connection
        self.cursor = connection.cursor()
        self.tickers_dict = {}

    def create_table(self, ticker: str, reset: bool):
        # ToDo reset table if reset is True
        self.tickers_dict[ticker] = ticker if not ticker.startswith("1INCH") else "INCH" + ticker[5:]
        self.cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {self.tickers_dict[ticker]} "
            f"(timestamp TIMESTAMP PRIMARY KEY, "
            f"open FLOAT, "
            f"high FLOAT, "
            f"low FLOAT, "
            f"close FLOAT, "
            f"volume FLOAT, "
            f"close_time TIMESTAMP, "
            f"quote_asset_volume FLOAT, "
            f"number_of_trades INTEGER, "
            f"taker_buy_base_asset_volume FLOAT, "
            f"taker_buy_quote_asset_volume FLOAT)"
        )

    def add_line_to_database(self, d_dict, key, close_db=False):
        timestamp = d_dict['timestamp']
        open_price = d_dict['open']
        high = d_dict['high']
        low = d_dict['low']
        close = d_dict['close']
        volume = d_dict['volume']
        close_time = d_dict['close_time']
        quote_asset_volume = d_dict['quote_asset_volume']
        number_of_trades = d_dict['number_of_trades']
        taker_buy_base_asset_volume = d_dict['taker_buy_base_asset_volume']
        taker_buy_quote_asset_volume = d_dict['taker_buy_quote_asset_volume']

        self.cursor.execute(
            f"REPLACE INTO {key} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp, open_price, high, low, close, volume, close_time, quote_asset_volume,
                number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume
            )
        )

        if close_db:
            self.connection.commit()
            self.connection.close()

    def initialize_with_historical_json(self, csv_file, reset: bool = True):
        super().initialize_with_historical_json(csv_file, reset)
        self.connection.commit()

    def _load_symbol_from_json(self, ticker: str, row: dict):
        self.add_line_to_database(row, self.tickers_dict[ticker], close_db=False)

    def get_all_tables(self):
        self.cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'"
        )
        table_names = self.cursor.fetchall()
        return table_names

    def get_data_frame_from_ticker(self, ticker):
        self.cursor.execute(f"SELECT * FROM {ticker};")
        df = pd.DataFrame(
            self.cursor.fetchall(),
            columns=[desc[0] for desc in self.cursor.description]
        ).set_index('timestamp')
        return df

    def close(self):
        self.connection.close()
