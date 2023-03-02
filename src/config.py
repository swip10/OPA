from pathlib import Path
import configparser

config = configparser.ConfigParser()
config.read(Path(__file__).parents[1].absolute() / "config" / "config.ini")
BINANCE_API_KEY = config['API']['BINANCE_API_KEY']
BINANCE_API_SECRET = config['API']['BINANCE_API_SECRET']
