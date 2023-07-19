from src.db.postgres import Postgres
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
plt.style.use("fivethirtyeight")
import warnings
import itertools
from tqdm import tqdm
warnings.filterwarnings("ignore")

# On instancie la classe Postgres
postgres = Postgres()

# On récupère le nom des différentes tables de la base de données, ainsi que le DataFrame pour le ticker choisi (nom ou position dans la liste obtenue)
table_names = postgres.get_all_table_names()
ticker = 'btcusdt'

df_ticker = postgres.get_table_as_dataframe(ticker)

# On récupère uniquement les données de la colonne 'close' et 'timestamp' pour notre modèle avec la colonne 'timestamp' comme index du DataFrame
df_LM = df_ticker[['timestamp', 'close']].set_index('timestamp')

# Tracé des données avec les années sur l'axe des abscisses
fig, ax = plt.subplots()
df_LM.plot(ax=ax)

# Obtention des positions des ticks pour chaque début d'année
year_starts = pd.date_range(start=df_LM.index[0], end=df_LM.index[-1], freq='YS')

# Configuration des positions des ticks pour les débuts d'année
ax.set_xticks(year_starts)

# Configuration des labels personnalisés pour afficher uniquement les années
ax.set_xticklabels(year_starts.strftime('%Y'))

plt.show()

#on sépare notre jeu de donnée :
to_row = int(len(df_LM)*0.9)

training_data = list(df_LM[:to_row]["close"])

testing_data = list(df_LM[to_row:]['close'])

#on instancie le modèle :

model_predictions =[]

n_test_observer = len(testing_data) #on instancie le nombre d'observations pour le set de test

#cela permet d'entrainer le modèle et de prevoir une ou plusieurs valeur en avance
for i in tqdm(range(n_test_observer), desc="Progression"):
    model = ARIMA(training_data, order=(4, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    actual_test_value = testing_data[i]
    training_data.append(actual_test_value)

print(model_fit.summary())

plt.figure(figsize=(15,9))
plt.grid(True)
date_range= df_LM[to_row:].index
plt.plot(date_range, model_predictions, color= 'blue',marker ='o',linestyle = 'dashed', label = 'BTC predicted price')
plt.plot(date_range, testing_data, color= 'red',label = 'BTC actual price')
plt.title('BTC price prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

mape = np.mean(np.abs(np.array(model_predictions[:len(testing_data)]) - np.array(testing_data)) / np.abs(np.array(testing_data)))

print("Mean Absolute Percentage Error = ", str(mape))

#on essaie de faire la même chose mais pour prédire la future évolution du cours du btc
# Réinitialisation de training_data pour contenir l'intégralité de df_LM
training_data_tot = list(df_LM["close"])

# On instancie le modèle ARIMA
model = ARIMA(training_data_tot, order=(4, 1, 0))
model_fit = model.fit()

# Prédictions sur l'ensemble futur de données (dates futures)
future_dates = pd.date_range(start=df_LM.index[-1], periods=99, freq='D')  # Remplacez 10 par le nombre de jours futurs que vous souhaitez prédire
future_predictions = []

for date in tqdm(future_dates, desc="Prédiction des valeurs futures"):
    output = model_fit.forecast()
    yhat = output[0]
    future_predictions.append(yhat)
    training_data_tot.append(yhat)  # Met à jour les données d'entraînement pour la prochaine prédiction

# Affichage du graphique des prédictions pour l'ensemble des données futures
plt.figure(figsize=(15, 9))
plt.grid(True)
plt.plot(future_dates, future_predictions, color='blue', marker='o', linestyle='dashed', label='Future BTC predicted price')
plt.title('Future BTC price prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()