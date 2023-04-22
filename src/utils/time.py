from datetime import datetime


def convert_ms_to_timestamp(time_in_ms: int):
    return datetime.fromtimestamp(time_in_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
