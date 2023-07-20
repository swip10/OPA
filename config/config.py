from pathlib import Path
import configparser

config = configparser.RawConfigParser()
config.read(Path(__file__).parents[1].absolute() / "config" / "config.ini")
BINANCE_API_KEY = config['API']['BINANCE_API_KEY']
BINANCE_API_SECRET = config['API']['BINANCE_API_SECRET']

# C'est ici que l'on choisi la liste des tickers que l'on veux appeler dans le script stream
EXTRACT_TICKERS = ['ETHBTC']

# C'est ici que l'on récupère le chemin local du JSON de données hist
CHEMIN_JSON_LOCAL = config['JSON']['CHEMIN_JSON']

def get_tickers(client):
    tickers = client.get_symbol_ticker()
    tickers = [ticker["symbol"] for ticker in tickers]
    for ticker in EXTRACT_TICKERS:
        if ticker not in tickers:
            raise KeyError(f"Ticker name {ticker} not available in Binance")
    return EXTRACT_TICKERS


host = config['SQL']['host']
port = config["SQL"]["port"]
database = config["SQL"]["database"]
db_user = config["SQL"]["db_user"]
db_password = config["SQL"]["db_password"]

try:
    twitterConsumerKey = config["TWITTER"].get("consumerKey", "")
    twitterConsumerSecret = config["TWITTER"].get("consumerSecret", "")
    twitterAccessToken = config["TWITTER"].get("accessToken", "")
    twitterAccessTokenSecret = config["TWITTER"].get("accessTokenSecret", "")
    twitterBearerToken = config["TWITTER"].get("bearerToken", "")
except KeyError:
    twitterConsumerKey = ""
    twitterConsumerSecret = ""
    twitterAccessToken = ""
    twitterAccessTokenSecret = ""
    twitterBearerToken = ""
