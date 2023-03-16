from binance.streams import ThreadedWebsocketManager
from binance.client import Client
from datetime import datetime
from time import sleep
import config
import sql_client

client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
tickers = config.get_tickers(client)

twm = ThreadedWebsocketManager()
# start is required to initialise its internal loop
twm.start()


def transform_and_append_database(msg):
    connector = sql_client.get_db_client()
    print(f"message type: {msg['e']}")
    kline = msg["k"]
    data = dict()
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
    print(data)
    sql_client.add_line_to_database(data, kline["s"], connector, close_db=True)


# example of function to call for opening a steam - todo define one handler for each function
# twm.start_depth_socket(callback=handle_socket_message, symbol=symbol)
for ticker in tickers:
    twm.start_kline_socket(callback=transform_and_append_database, symbol=ticker, interval='1s')

# twm.start_aggtrade_socket(callback=handle_socket_message, symbol=symbol)

sleep(5)
twm.stop()