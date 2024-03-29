import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.db.postgres import Postgres
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Rediriger les avertissements et les messages de fréquence inconnue vers stderr
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

postgres = Postgres()

table_names = postgres.get_all_table_names()
ticker = 'btcusdt'

df_ticker = postgres.get_table_as_dataframe(ticker)

df_LM = df_ticker[['timestamp', 'close']]

# Convertir la colonne 'timestamp' en un objet datetime si elle n'est pas déjà en datetime
df_LM['timestamp'] = pd.to_datetime(df_LM['timestamp'])

# Trier les données par date
df_LM = df_LM.sort_values(by='timestamp')

# Définir l'index du DataFrame comme la colonne 'timestamp'
df_LM.set_index('timestamp', inplace=True)

# Initialisation des prédictions
forecast_steps = 90
forecast = []

# Prédire les 90 prochains intervalles de 8 heures (30 jours)
for i in tqdm(range(forecast_steps), desc="Prédiction en cours"):
    # Entraîner un modèle SVR à chaque itération
    model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    
    # Préparer les données d'entraînement
    X_train = np.arange(len(df_LM)).reshape(-1, 1)
    y_train = df_LM['close']
    
    # Normaliser les données d'entraînement
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    
    # Adapter le modèle aux données d'entraînement
    model.fit(X_train, y_train.ravel())
    
    # Prédire pour le prochain intervalle de 8 heures
    next_X = np.array([[len(df_LM)]])
    next_X = scaler_X.transform(next_X)
    next_forecast = scaler_y.inverse_transform(model.predict(next_X).reshape(-1, 1))[0][0]
    
    # Ajouter la prédiction à la liste
    forecast.append(next_forecast)
    
    # Mettre à jour les données d'entraînement avec la dernière prédiction
    df_LM.loc[df_LM.index[-1] + pd.Timedelta(hours=8)] = next_forecast

# Créer une nouvelle colonne 'predictions' dans le DataFrame et y stocker les prédictions
df_LM['predictions'] = np.nan
df_LM['predictions'].iloc[-forecast_steps:] = forecast

# Créer un nouvel index pour les prédictions à la suite des données passées
index_predictions = pd.date_range(start=df_LM.index[-1] + pd.Timedelta(hours=8), 
                                 periods=forecast_steps, freq='8H')
df_predictions = pd.DataFrame(forecast, index=index_predictions, columns=['predictions'])

# Concaténer les données passées et les prédictions
df_combined = pd.concat([df_LM, df_predictions])

# Afficher le graphique des cours passés et des prédictions à la suite
plt.figure(figsize=(12, 6))
plt.plot(df_combined.index, df_combined['close'], label='Cours passé', linewidth=2)
plt.plot(df_combined.index, df_combined['predictions'], color='red', label='Prédictions')
plt.title('Prédictions du cours du Bitcoin avec SVR')
plt.xlabel('Date')
plt.ylabel('Prix du Bitcoin')
plt.legend()
plt.show()
