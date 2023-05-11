from binance.streams import ThreadedWebsocketManager
from binance.client import Client
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

# twm.start_depth_socket(callback=handle_socket_message, symbol=symbol)
for ticker in tickers:
    twm.start_kline_socket(callback=client.callback_stream_msg, symbol=ticker, interval='1s')

# twm.start_aggtrade_socket(callback=handle_socket_message, symbol=symbol)

sleep(5)

twm.stop()
twm.join()

client.pprint_one_document_in_collection(Collection.KLINES)
client.close()
