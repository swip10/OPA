from src.db.postgres import Postgres
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
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

#On va décomposer notre courbe de données en 3 parties, une partie trend, une partie seasonal et un résidu afin de voir si notre Time serie est stationnaire ou non :

decomp = sm.tsa.seasonal_decompose(df_LM['close'], model='additive')
decomp.plot()
plt.show()

adftest = adfuller(df_LM['close'])

print('p-value of adfullertest =', adftest[1]) #si pvalue > 0.05 --> non stationnaire


#on sépare notre jeu de donnée :

train = df_LM.iloc[:int(0.8 * len(df_LM))]

test = df_LM.iloc[int(0.8 * len(df_LM)):]


#on instancie le modèle ainsi que les hyperparamètres
p = range(0,10)
d = range(0,10)
q = range(0,4)

pdq_combination = list(itertools.product(p, d, q)) #4,2,3

# Désactiver l'affichage des avertissements de convergence
sm.tsa.statespace.SARIMAX.display_convergence_warnings = False

# Initialisation des variables pour suivre la RMSE minimale et les paramètres correspondants
min_rmse = float('inf')  # Valeur initiale définie à l'infini
best_order = None

# Utilisation de tqdm pour afficher une barre de progression
# with tqdm(total=len(pdq_combination)) as progress_bar:
#     for pdq in pdq_combination:
#         try:
#             model = ARIMA(train, order=pdq).fit()
#             pred = model.predict(start=len(train), end=(len(df_LM)-1))
#             error = np.sqrt(mean_squared_error(test, pred))
#             if error < min_rmse:
#                 min_rmse = error
#                 best_order = pdq
#         except:
#             continue
        
#         # Mise à jour de la barre de progression
#         progress_bar.update(1)


#print("Best order (p, d, q):", best_order)
#print("RMSE:", min_rmse)

best_model = ARIMA(df_LM['close'], order=(4,2,3)).fit()

#on instancie le nombre de jours/8h qu'on veut prédire
nbre_future_days = 300

future_pred = best_model.predict(len(df_LM), len(df_LM)+ nbre_future_days)

#visualisation :

df_LM.plot(legend=True, label ='Data hist')
future_pred.plot(legend=True, label='Prédictions')
plt.show()
