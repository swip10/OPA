import pandas as pd
from config import config
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager


client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
tickers = client.get_all_tickers()
print(tickers)


tickers_df = pd.DataFrame(tickers)
tickers_df.head()

tickers_df.set_index('symbol', inplace=True)

tickers_df.head()
