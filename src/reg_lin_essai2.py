from src.db.postgres import Postgres
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

# Instanciation de la classe Postgres
postgres = Postgres()

# Récupération du nom des différentes tables de la base de données, ainsi que le df pour le ticker choisi (nom ou position dans la liste obtenue)
table_names = postgres.get_all_table_names()
ticker = 'btcusdt'

df_ticker = postgres.get_table_as_dataframe(ticker)

# Récupération uniquement des données de la colonne close et timestamp pour notre modèle
df_LM = df_ticker[['timestamp', 'close']]

# Transfert des valeurs de close dans un array numpy
dataset = df_LM.filter(['close']).values

training_data_len = math.ceil(len(dataset) * 0.8)

# Mise à l'échelle des données
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Création du dataset d'entraînement
train_data = scaled_data[0:training_data_len, :]

# Séparation des données en ensembles x_train et y_train
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

# Conversion en tableaux NumPy
x_train, y_train = np.array(x_train), np.array(y_train)

# Création du modèle de régression linéaire
model = LinearRegression()

# Entraînement du modèle
model.fit(x_train, y_train)

# Création de l'array contenant les valeurs pour le dataset de test
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# Conversion en un array numpy
x_test = np.array(x_test)

# Prédiction avec le modèle de régression linéaire
predictions = model.predict(x_test)

# Inversion de la mise à l'échelle des prédictions
predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

#Récupération de la RMSE(root mean squarred error )
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
accuracy = 1 - np.mean(np.abs(predictions - y_test) / y_test)
print("Erreur quadratique moyenne:", rmse)
print("Précision:", accuracy)

# Visualisation des données et des prédictions
train = df_LM[:training_data_len]
valid = df_LM[training_data_len:]

valid_copy = valid.copy()
valid_copy['Predictions'] = predictions

plt.title('Modèle de régression linéaire')
plt.xlabel('Date')
plt.ylabel('Close Price (USD$)')
plt.plot(train['close'])
plt.plot(valid_copy[['close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper right')
plt.show()
