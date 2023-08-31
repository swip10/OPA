from pathlib import Path
import configparser

config = configparser.RawConfigParser()
config.read(Path(__file__).parents[1].absolute() / "config" / "config.ini")
BINANCE_API_KEY = config['API']['BINANCE_API_KEY']
BINANCE_API_SECRET = config['API']['BINANCE_API_SECRET']

# C'est ici que l'on choisi la liste des tickers que l'on veux appeler dans le script stream
EXTRACT_TICKERS = ['ETHBTC']

# C'est ici que l'on récupère le chemin local du JSON de données hist
CHEMIN_JSON_MACHINE = config['JSON']['CHEMIN_JSON_LOCAL']
CHEMIN_JSON_IMAGE = config['JSON']['CHEMIN_JSON_IMAGE']
if Path(CHEMIN_JSON_MACHINE).exists():
    CHEMIN_JSON_LOCAL = CHEMIN_JSON_MACHINE
else:
    CHEMIN_JSON_LOCAL = CHEMIN_JSON_IMAGE


def get_tickers(client):
    tickers = client.get_symbol_ticker()
    tickers = [ticker["symbol"] for ticker in tickers]
    for ticker in EXTRACT_TICKERS:
        if ticker not in tickers:
            raise KeyError(f"Ticker name {ticker} not available in Binance")
    return EXTRACT_TICKERS

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


config_sql = configparser.RawConfigParser()
with open(Path(__file__).parents[1].absolute() / "config" / "config_sql.ini") as stream:
    config_sql.read_string("[SERVICES]\n" + stream.read())  # This line does the trick.

host = config_sql['SERVICES']['POSTGRES_HOST']
port = config_sql['SERVICES']["POSTGRES_PORT"]
database = config_sql['SERVICES']["POSTGRES_DB"]
db_user = config_sql['SERVICES']["POSTGRES_USER"]
db_password = config_sql['SERVICES']["POSTGRES_PASSWORD"]

mongodb_host = config_sql['SERVICES']['MONGODB_HOST']
mongodb_port = int(config_sql['SERVICES']["MONGODB_PORT"])
