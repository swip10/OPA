import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.db.postgres import Postgres
from tqdm import tqdm 
import sys
import warnings

# Rediriger les avertissements et les messages de fréquence inconnue vers stderr
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

# Vérifier si la série temporelle est stationnaire (test de Dickey-Fuller)
result = adfuller(df_LM['close'])
print('Test de Dickey-Fuller:')
print(f'Statistic: {result[0]}')
print(f'p-value: {result[1]}')
print(f'Critial Values:')
for key, value in result[4].items():
    print(f'   {key}: {value}')

# Si la série n'est pas stationnaire, effectuer une différenciation pour la rendre stationnaire
if result[1] > 0.05:
    df_LM_diff = df_LM['close'].diff().dropna()
else:
    df_LM_diff = df_LM['close']

# Tracer les autocorrélations et autocorrélations partielles pour déterminer les ordres AR et MA
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
#plot_acf(df_LM_diff, lags=40, ax=ax1)
#plot_pacf(df_LM_diff, lags=40, ax=ax2)
#plt.show()

# Initialisation des prédictions
forecast_steps = 90
forecast = []

# Prédire les 90 prochains intervalles de 8 heures (30 jours)
for i in tqdm(range(forecast_steps), desc="Prédiction en cours"):
    # Entraîner un modèle ARIMA à chaque itération
    order = (3, 1, 3)  # Utilisation de p=3 et q=3
    model = sm.tsa.ARIMA(df_LM['close'], order=order)
    results = model.fit()
    
    # Faire une prédiction
    next_forecast = results.forecast(steps=1, alpha=0.05)[0]
    
    # Ajouter la prédiction à la liste
    forecast.append(next_forecast)
    
    # Mettre à jour les données d'entraînement avec la dernière prédiction
    df_LM.loc[df_LM.index[-1] + pd.Timedelta(hours=8)] = next_forecast
    df_LM_diff = df_LM['close'].diff().dropna()

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
plt.title('Prédictions du cours du Bitcoin avec ARIMA')
plt.xlabel('Date')
plt.ylabel('Prix du Bitcoin')
plt.legend()
plt.show()
