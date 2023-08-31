from config import config
from src.db.mongodb import MongoOPA, Collection


client = MongoOPA(
    host=config.mongodb_host,
    port=config.mongodb_port
)

client.initialize_with_historical_json("../data/ticker_data_hist.json")
client.pprint_one_document_in_collection(Collection.KLINES)

client.close()
