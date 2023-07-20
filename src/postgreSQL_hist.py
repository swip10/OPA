from config.config import CHEMIN_JSON_LOCAL
from src.db.postgres import Postgres


postgres_client = Postgres()

postgres_client.initialize_with_historical_json(CHEMIN_JSON_LOCAL, reset=True)

postgres_client.close()
