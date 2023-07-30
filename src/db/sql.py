import abc

import pandas as pd
from typing import List
from src.db.clientdb import DBClient
from src.utils.time import convert_ms_to_timestamp


class SQL(DBClient):

    def __init__(self, connection):
        super().__init__()
        self.connection = connection
        self.cursor = connection.cursor()
        self.tickers_dict = {}

    def create_table(self, ticker: str, reset: bool) -> None:
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
            f"taker_buy_quote_asset_volume FLOAT, "
            # ajout de la colonne pour connaître la provenance de la donnée (true = stream, false = hist)
            f"data_origin BOOLEAN DEFAULT True)"
        )

    def create_tables(self, tickers: List[str], reset: bool = False) -> None:
        for ticker in tickers:
            self.create_table(ticker, reset)

    def add_line_to_database(self, d_dict: dict, key: str, close_db: bool = False) -> None:
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
        # Recupération de la données data_origin du dictionnaire (soit en provenance du JSON hist, soit du callback_stream_msg)
        data_origin = d_dict['data_origin']

        self.cursor.execute(
            f"INSERT INTO {key} VALUES ('{timestamp}', {open_price}, {high}, {low}, {close}, {volume}, '{close_time}', "
            f"{quote_asset_volume}, {number_of_trades}, {taker_buy_base_asset_volume}, {taker_buy_quote_asset_volume}, {data_origin}) ON CONFLICT (timestamp) DO NOTHING"
        )

        if close_db:
            self.connection.commit()
            self.connection.close()

    def initialize_with_historical_json(self, csv_file, reset: bool = True) -> None:
        super().initialize_with_historical_json(csv_file, reset)
        self.connection.commit()

    def _load_symbol_from_json(self, ticker: str, row: dict) -> None:
        if 'data_origin' not in row:
            row['data_origin'] = False
        self.add_line_to_database(row, self.tickers_dict[ticker], close_db=False)

    @abc.abstractmethod
    def get_all_table_names(self) -> List[str]:
        pass

    def get_data_frame_from_ticker(self, ticker: str) -> pd.DataFrame:
        self.cursor.execute(f"SELECT * FROM {ticker};")
        df = pd.DataFrame(
            self.cursor.fetchall(),
            columns=[desc[0] for desc in self.cursor.description]
        ).set_index('timestamp')
        return df

    def execute_pandas_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.connection)

    def callback_stream_msg(self, msg: dict) -> None:
        kline = msg["k"]
        data = dict()
        data["timestamp"] = convert_ms_to_timestamp(kline["t"])
        data["close_time"] = convert_ms_to_timestamp(kline["T"])
        data["open"] = kline["o"]
        data["high"] = kline["h"]
        data["low"] = kline["l"]
        data["close"] = kline["c"]
        data["volume"] = kline["v"]
        data["quote_asset_volume"] = kline["q"]
        data["number_of_trades"] = kline["n"]
        data["taker_buy_base_asset_volume"] = kline["V"]
        data["taker_buy_quote_asset_volume"] = kline["Q"]
        # Création de la colonne data_origin à True car nous sommes dans la partie stream
        data["data_origin"] = True
        print(data)
        self.add_line_to_database(data, self.tickers_dict[kline["s"]], close_db=False)
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()
