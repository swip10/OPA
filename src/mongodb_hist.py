import json
from mongodb import MongoOPA, Collection


client = MongoOPA(
    host="127.0.0.1",
    port=27017
)

client.create_collections(reset=True)

with open("../ticker_data_hist.json", "r") as json_file:
    hist_data = json.load(json_file)

for symbol, data in hist_data.items():
    for line in data:
        line["symbol"] = symbol
    client.insert_documents_to_collection(data, Collection.KLINES)

client.pprint_one_document_in_collection(Collection.KLINES)

