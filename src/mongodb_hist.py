from src.db.mongodb import MongoOPA, Collection


client = MongoOPA(
    host="127.0.0.1",
    port=27017
)

client.initialize_with_historical_json("../data/ticker_data_hist.json")
client.pprint_one_document_in_collection(Collection.KLINES)

client.close()
