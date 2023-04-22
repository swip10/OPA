from datetime import datetime

import pandas as pd

FORMAT = '%Y-%m-%d %H:%M:%S'


def convert_ms_to_timestamp(time_in_ms: int):
    return datetime.fromtimestamp(time_in_ms / 1000).strftime(FORMAT)


def _convert_timestamp_to_ms(timestamp):
    dt_obj = datetime.strptime(timestamp, FORMAT)
    return dt_obj.timestamp() * 1000


def convert_timestamp_to_ms(timestamp):
    if isinstance(timestamp, pd.Index):
        res = []
        for time in timestamp:
            res.append(_convert_timestamp_to_ms(time))
        return pd.Series(res)
    else:
        return _convert_timestamp_to_ms(timestamp)
