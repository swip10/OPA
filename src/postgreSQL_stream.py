from binance.streams import ThreadedWebsocketManager
from binance.client import Client
from time import sleep
from config import config
from src.db.postgres import Postgres


client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
tickers = config.get_tickers(client)
print(tickers)

twm = ThreadedWebsocketManager()
# start is required to initialise its internal loop
twm.start()

postgres_client = Postgres()
# update 24_05 : reset --> True
postgres_client.create_tables(tickers, reset=True)

for ticker in tickers:
    twm.start_kline_socket(callback=lambda msg: postgres_client.callback_stream_msg(msg, ticker), symbol=ticker, interval='1s')
 
sleep(5)

twm.stop()
twm.join()
postgres_client.close()
