from binance.streams import ThreadedWebsocketManager
from binance.client import Client
from datetime import datetime
from time import sleep
from config import config
from src.db.mongodb import MongoOPA, Collection


client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
tickers = config.get_tickers(client)

twm = ThreadedWebsocketManager()
twm.start()

client = MongoOPA(
    host="127.0.0.1",
    port=27017
)
client.create_collections(reset=True)


def transform_and_append_database(msg):
    kline = msg["k"]
    data = dict()
    data["symbol"] = kline["s"]
    data["timestamp"] = datetime.fromtimestamp(kline["t"]/1000).strftime('%Y-%m-%d %H:%M:%S')
    data["close_time"] = datetime.fromtimestamp(kline["T"]/1000).strftime('%Y-%m-%d %H:%M:%S')
    data["open"] = kline["o"]
    data["high"] = kline["h"]
    data["low"] = kline["l"]
    data["close"] = kline["c"]
    data["volume"] = kline["v"]
    data["quote_asset_volume"] = kline["q"]
    data["number_of_trades"] = kline["n"]
    data["taker_buy_base_asset_volume"] = kline["V"]
    data["taker_buy_quote_asset_volume"] = kline["Q"]
    client.insert_document_to_collection(doc=data, collection_name=Collection.KLINES)


# twm.start_depth_socket(callback=handle_socket_message, symbol=symbol)
for ticker in tickers:
    twm.start_kline_socket(callback=transform_and_append_database, symbol=ticker, interval='1s')

# twm.start_aggtrade_socket(callback=handle_socket_message, symbol=symbol)

sleep(5)

twm.stop()
twm.join()

client.pprint_one_document_in_collection(Collection.KLINES)
client.close()
