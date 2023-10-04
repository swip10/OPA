import os
from binance.streams import ThreadedWebsocketManager
from binance.client import Client
from time import sleep
from config import config
from src.db.postgres import Postgres
from datetime import datetime

def launch_stream(ticker):
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    twm = ThreadedWebsocketManager()
    twm.start()

    postgres_client = Postgres()
    postgres_client.create_tables(ticker, reset=False)

    write_start_time_to_file()
    twm.start_kline_socket(callback=lambda msg: postgres_client.callback_stream_msg(msg, ticker), symbol=ticker, interval='1s')


    while os.path.exists('start_streaming.txt') and check_file_content()[1] == ticker and check_file_content()[0] == 'start':
        sleep(1)

    twm.stop()
    twm.join()
    postgres_client.close()

def check_file_content():
    """Read the content of the file and return the first two lines as a tuple."""
    if not os.path.exists('start_streaming.txt'):
        return "stop", ""

    with open('start_streaming.txt', 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
        return first_line, second_line


def write_start_time_to_file():
    # Lire le contenu actuel du fichier
    with open('start_streaming.txt', 'r') as f:
        lines = f.readlines()

    # Si le fichier contient déjà 3 lignes, remplacez simplement la troisième ligne.
    # Sinon, ajoutez une nouvelle troisième ligne.
    formatted_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if len(lines) >= 3:
        lines[2] = formatted_date_time + '\n'
    else:
        lines.append(formatted_date_time + '\n')

    # Écrire le contenu mis à jour au fichier
    with open('start_streaming.txt', 'w') as f:
        f.writelines(lines)



if __name__ == "__main__":
    while True:
        command, ticker = check_file_content()
        if os.path.exists('start_streaming.txt') and command == 'start':
            launch_stream(ticker)
        sleep(1)
