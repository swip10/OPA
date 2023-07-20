from src.db.postgres import Postgres
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

#On instancie la classe Postgres

postgres=Postgres()

# On récupère le nom des différentes tables de la base de donnée, ainsi que le df pour le ticker choisi (nom, ou position dans la liste obtenue)
table_names=postgres.get_all_table_names()
ticker='btcusdt'

df_ticker = postgres.get_table_as_dataframe(ticker)

#On récupère uniquement les datas de la colonne close et timestamp pour notre modèle


# Tracer un graphique avec les données de la colonne "close"
fig = plt.plot(df_ticker['timestamp'], df_ticker['close'])
plt.title(f"Close price for {ticker} over the last 1k days")
plt.xlabel("Date")
plt.ylabel("Close Price USD ($)")

# Afficher le graphique
plt.show()

# on sépare les variables catégorielles et la variable cible

x= df_ticker.drop(['close', 'close_time'], axis = 1)
x.set_index(['timestamp'], inplace = True)


y= df_ticker['close']


#on divise les données en jeu d'entrainement et de test : 

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

#On standardise les valeurs : 
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

#on instancie le modèle : 

lr =LinearRegression()

lr.fit(X_train_scaled,y_train)

predictions = lr.predict(X_test_scaled)

rmse = mean_squared_error(y_test, predictions, squared=False)
print('RMSE:', rmse)

#affichage du score du modèle : 


print(lr.score(X_train_scaled,y_train))

print(lr.score(X_test_scaled,y_test))
 
#affichage du graphe montrant les prédictions obtenues : 


plt.figure(figsize=(10, 6))  

plt.scatter(X_train.index, y_train, color='blue', label='Training Data')

plt.scatter(X_test.index, y_test, color='green', label='Test Data')

plt.scatter(X_test.index, predictions, color='red', label='Predictions')

plt.title(f"Close Price for {ticker}")
plt.xlabel("Timestamp")
plt.ylabel("Close Price (USD)")
plt.legend()

plt.show()


