from enum import Enum
from pprint import pprint
from pymongo import MongoClient
from typing import (
    Optional,
    Sequence,
    Union,
    Dict,
)


class Collection(Enum):
    KLINES = "klines"


class MongoOPA(MongoClient):
    MongoClient.HOST = "127.0.0.1"
    MongoClient.PORT = "27017"

    def __init__(
        self,
        host: Optional[Union[str, Sequence[str]]] = None,
        port: Optional[int] = None
    ) -> None:
        super().__init__(host, port)
        self.db_name = "opa"
        self.collection_names = [Collection.KLINES]

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
            self.opa_db.create_collection(name=str(collection_name))

    def insert_document_to_collection(self, doc: Dict, collection_name: Collection) -> None:
        self.opa_db[str(collection_name)].insert_one(doc)

    def insert_documents_to_collection(self, docs, collection_name: Collection) -> None:
        self.opa_db[str(collection_name)].insert_many(docs)

    def pprint_one_document_in_collection(self, collection_name: Collection) -> None:
        pprint(self.opa_db[str(collection_name)].find_one())
