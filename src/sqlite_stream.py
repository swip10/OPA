from binance.streams import ThreadedWebsocketManager
from binance.client import Client
from time import sleep
from config import config
from src.db.sqlite import SQLiteOPA


client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
tickers = config.get_tickers(client)

twm = ThreadedWebsocketManager()
# start is required to initialise its internal loop
twm.start()

sqlite_client = SQLiteOPA(check_same_thread=False)
sqlite_client.create_tables(tickers, reset=False)

for ticker in tickers:
    twm.start_kline_socket(callback=sqlite_client.callback_stream_msg, symbol=ticker, interval='1s')

sleep(5)

twm.stop()
twm.join()
sqlite_client.close()
