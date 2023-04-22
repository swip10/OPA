from db.sqlite import SQLiteOPA


sqlite_client = SQLiteOPA()
sqlite_client.initialize_with_historical_json('../data/ticker_data_hist.json', reset=True)

sqlite_client.close()
