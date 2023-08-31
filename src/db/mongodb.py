from abc import ABC
from enum import Enum
from pprint import pprint

import pandas as pd

from config import config
from src.db.clientdb import DBClient
from pymongo import MongoClient
from src.utils.time import convert_ms_to_timestamp
from typing import (
    Optional,
    Sequence,
    Union,
    Dict,
)


class Collection(Enum):
    KLINES = "klines"
    WIKI = "wiki"


class MongoOPA(MongoClient, DBClient):
    MongoClient.HOST = config.mongodb_host
    MongoClient.PORT = config.mongodb_port

    def __init__(
        self,
        host: Optional[Union[str, Sequence[str]]] = None,
        port: Optional[int] = None
    ) -> None:
        super().__init__(host, port)
        self.db_name = "opa"
        self.collection_names = [Collection.KLINES, Collection.WIKI]

    @property
    def opa_db(self):
        return self[self.db_name]

    def reset_collections(self) -> None:
        for collection_name in self.collection_names:
            if str(collection_name) in self.opa_db.list_collection_names():
                self.opa_db.drop_collection(str(collection_name))

    def create_collections(self, reset: bool = False) -> None:
        """
            Create all the collections defined in the class constructor 'collection_names'
        """
        if reset:
            self.reset_collections()
        for collection_name in self.collection_names:
            if str(collection_name) not in self.opa_db.list_collection_names():
                self.opa_db.create_collection(name=str(collection_name))

    def insert_document_to_collection(self, doc: Dict, collection_name: Collection) -> None:
        self.opa_db[str(collection_name)].insert_one(doc)

    def insert_documents_to_collection(self, docs, collection_name: Collection) -> None:
        if len(docs) != 0:
            self.opa_db[str(collection_name)].insert_many(docs)

    def pprint_one_document_in_collection(self, collection_name: Collection) -> None:
        pprint(self.opa_db[str(collection_name)].find_one())

    def initialize_with_historical_json(self, csv_file, reset: bool = True):
        self.create_collections(reset)
        super().initialize_with_historical_json(csv_file, reset)

    def initialize_with_wiki_revisions(self, editions, reset: bool = False):
        self.create_collections(reset)
        self.insert_documents_to_collection(editions, Collection.WIKI)

    def get_wiki_last_revision(self):
        res = self.opa_db[str(Collection.WIKI)].aggregate([
          {"$project": { "maxId": {"$max": "$revid" } }},
        ])
        all_max = [line["maxId"] for line in res]
        if len(all_max) == 0:
            return 1054294052
        else:
            return max(all_max)

    def get_average_sentiment_over_time(self) -> pd.DataFrame:
        res = self.opa_db[str(Collection.WIKI)].aggregate([
            {"$project": {"_id": "$id", "timestamp": "$timestamp", "average_sentiment": {"$avg": "$sentiments"}}},
        ])
        return pd.DataFrame(res)

    def _load_symbol_from_json(self, ticker: str, row: dict):
        row["symbol"] = ticker
        self.insert_document_to_collection(row, Collection.KLINES)

    def callback_stream_msg(self, msg):
        kline = msg["k"]
        data = dict()
        data["symbol"] = kline["s"]
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
        self.insert_document_to_collection(doc=data, collection_name=Collection.KLINES)
