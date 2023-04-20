from pathlib import Path
import configparser

config = configparser.ConfigParser()
config.read(Path(__file__).parents[1].absolute() / "config" / "config.ini")
BINANCE_API_KEY = config['API']['BINANCE_API_KEY']
BINANCE_API_SECRET = config['API']['BINANCE_API_SECRET']

EXTRACT_TICKERS = ['BTCEUR', 'ETHEUR', 'BNBEUR', 'XRPEUR']



def get_tickers(client):
    tickers = client.get_symbol_ticker()
    tickers = [ticker["symbol"] for ticker in tickers]
    for ticker in EXTRACT_TICKERS:
        if not ticker in tickers:
            raise KeyError(f"Ticker name {ticker} not available in Binance")
    return EXTRACT_TICKERS


host=config["SQL"]["host"]
port=config["SQL"]["port"]
database=config["SQL"]["database"]
db_user=config["SQL"]["db_user"]
db_password=config["SQL"]["db_password"]