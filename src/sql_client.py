import sqlite3


def get_db_client():
    return sqlite3.connect('BDD_hist.sqlite')


def add_line_to_database(d, key, conn, close_db=False):
    symbol = key
    timestamp = d['timestamp']
    open_price = d['open']
    high = d['high']
    low = d['low']
    close = d['close']
    volume = d['volume']
    close_time = d['close_time']
    quote_asset_volume = d['quote_asset_volume']
    number_of_trades = d['number_of_trades']
    taker_buy_base_asset_volume = d['taker_buy_base_asset_volume']
    taker_buy_quote_asset_volume = d['taker_buy_quote_asset_volume']

    cursor = conn.cursor()
    cursor.execute(f"REPLACE INTO {key} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (
              symbol, timestamp, open_price, high, low, close, volume, close_time, quote_asset_volume, number_of_trades,
              taker_buy_base_asset_volume, taker_buy_quote_asset_volume))

    if close_db:
        conn.commit()
        conn.close()


def create_table_database(cursor, key):
    cursor.execute(f'''CREATE TABLE IF NOT EXISTS {key}
                (symbol TEXT,
                 timestamp TEXT, 
                 open FLOAT, 
                 high FLOAT, 
                 low FLOAT, 
                 close FLOAT, 
                 volume FLOAT, 
                 close_time TEXT, 
                 quote_asset_volume FLOAT, 
                 number_of_trades INTEGER, 
                 taker_buy_base_asset_volume FLOAT, 
                 taker_buy_quote_asset_volume FLOAT, 
                 PRIMARY KEY (timestamp, symbol))''')
