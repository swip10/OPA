from binance.streams import ThreadedWebsocketManager
from time import sleep

symbol = 'BNBBTC'

twm = ThreadedWebsocketManager()
# start is required to initialise its internal loop
twm.start()


def handle_socket_message(msg):
    print(f"message type: {msg['e']}")
    print(msg)


# example of function to call for opening a steam - todo define one handler for each function
twm.start_depth_socket(callback=handle_socket_message, symbol=symbol)

twm.start_kline_socket(callback=handle_socket_message, symbol=symbol)

twm.start_aggtrade_socket(callback=handle_socket_message, symbol=symbol)

sleep(5)
twm.stop()
