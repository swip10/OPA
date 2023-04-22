import abc
import json
from tqdm import tqdm


class DBClient(metaclass=abc.ABCMeta):
    """
        Pure virtual class for DataBase interface
        https://stackoverflow.com/questions/26458618/are-python-pure-virtual-functions-possible-and-or-worth-it
    """

    def initialize_with_historical_json(self, csv_file, reset: bool = True):
        """
        Init from json file - this function could be multi-thread

        :param csv_file:
        :param reset:
        :return:
        """
        with open(csv_file, "r") as json_file:
            hist_data = json.load(json_file)
        for ticker, data in hist_data.items():
            self.create_table(ticker, reset)
            for row in tqdm(data, desc=f"Insert {ticker}"):
                self._load_symbol_from_json(ticker, row)

    @abc.abstractmethod
    def _load_symbol_from_json(self, ticker: str, row: dict):
        pass

    @abc.abstractmethod
    def callback_stream_msg(self, msg):
        pass

    def create_table(self, ticker, reset):
        pass
