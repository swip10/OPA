from src.db.postgres import Postgres


postgres_client = Postgres()

postgres_client.initialize_with_historical_json('../data/ticker_data_hist_new.json', reset=True)

postgres_client.close()
