import joblib
import numpy as np
import tensorflow as tf
from pydantic import BaseModel
from typing import Optional, List
from fastapi import FastAPI
from fastapi import Depends, HTTPException, status, Query
from keras.saving.saving_api import load_model
from pathlib import Path
from collections import namedtuple
from src.db.postgres import Postgres
from datetime import datetime

api = FastAPI(title="Price predictor",
              description="Predict the price for a currency as well as its volume powered by FastAPI.",
              version="0.0.1")

the_script = Path(__file__)
model = load_model(the_script.parents[2] / "models" / "001_close_volume" / "keras_next")
volume_scaler = joblib.load(the_script.parents[2] / "models" / "001_close_volume" / "scaler_volume.save")
price_scaler = joblib.load(the_script.parents[2] / "models" / "001_close_volume" / "scaler_close_price.save")


class History(BaseModel):
    price: List[float]
    volume: List[float]
    currency: Optional[str] = None


class Predictions(BaseModel):
    prices: List[float]
    volumes: List[float]


@api.post('/prediction', name='Prediction of the next price in x numbers of hours')
def post_prediction(item: History, next_hours: int = 8) -> Predictions:
    """Return the predicted price in the next number of hours
    """
    x = np.array([item.price, item.volume]).T
    if x.shape != (59, 2):
        raise HTTPException(
            status_code=status.HTTP_204_NO_CONTENT,
            detail="price and volume history have not the correct length - required to be 59"
        )

    nb_predictions = int(next_hours // 8)
    if nb_predictions == 0:
        raise HTTPException(
            status_code=status.HTTP_204_NO_CONTENT,
            detail="Ask for at least prediction for next 8 hours"
        )

    close = price_scaler.transform(x[:, 0].reshape(-1, 1))
    volume = volume_scaler.transform(x[:, 1].reshape(-1, 1))
    # x = np.expand_dims(x, axis=0)  # shape should be (1, 59, 2) because predicting one batch at a time
    x = np.array([close, volume]).T
    x = tf.convert_to_tensor(x, np.float32)
    predictions = []
    for i in range(0, nb_predictions):
        y = model.predict(x)[0]
        # remove first value and add the prediction before redoing the model's prediction
        x = tf.concat([x[:, 1:, :], [[[y[0], y[1]]]]], axis=1)
        predictions.append(y)
    predictions = np.array(predictions)
    prices = price_scaler.inverse_transform(predictions.T[0].reshape(-1, 1)).reshape(nb_predictions)
    volumes = volume_scaler.inverse_transform(predictions.T[0].reshape(-1, 1)).reshape(nb_predictions)
    return Predictions(prices=prices, volumes=volumes)


@api.get('/currency', name='Get model currency')
def get_currency():
    return {
        'currency': "ETHBTC"
    }

#on établi une connexion à la BDD Postgres
database = Postgres()

#méthode pour récupérer le dataframe d'un ticker spécifique via la bdd postgres

@api.get('/dfcurrency/{ticker}', name='Get Currency DataFrame')
def get_currency_dataframe(ticker: str):
    try:
        # On récupère la liste des noms de tables (tickers) disponibles dans la base de données
        available_tickers = database.get_all_table_names()
        
        # On vérifie si le ticker demandé est bien dans la liste
        if ticker not in [item[0] for item in available_tickers]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"DataFrame not found for {ticker}. Ticker not found in the database."
            )

        df = database.get_table_as_dataframe(ticker)

        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"DataFrame is empty for {ticker}."
            )

        
        df['timestamp'] = df['timestamp'].astype(int)

        # Récupère les X dernières lignes du DataFrame
        df_tail = df.tail(60)

        # Transforme la colonne 'timestamp' en format lisible (timestamp UNIX en date)
        df_tail['timestamp'] = df_tail['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

        return df_tail

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
