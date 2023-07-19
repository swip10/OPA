from src.db.postgres import Postgres
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

#On instancie la classe Postgres

postgres=Postgres()

# On récupère le nom des différentes tables de la base de donnée, ainsi que le df pour le ticker choisi (nom, ou position dans la liste obtenue)
table_names=postgres.get_all_table_names()
ticker='btcusdt'

df_ticker = postgres.get_table_as_dataframe(ticker).set_index('timestamp')

#on crée une variable indiquant le nombre de jour à l'avance dont on veut prédire le prix

future_days = 15

#on crée une nouvelle colonne qui est en fait la colonne close décalée du nombre de jours souhaités :

df_ticker[str(future_days)+ '_day_price_forecast']= df_ticker[['close']].shift(-future_days)

# On transfère les données de la colonne close dans un array numpy (ce sera nos features)

X = np.array(df_ticker[['close']])
X= X[:df_ticker.shape[0]-future_days]          #on ajuste la taille de l'array en fonction du nombre de jour dont on veut prédire le prix

# On fait de même pour la colonne future price, notre target

y = np.array(df_ticker[str(future_days)+ '_day_price_forecast'])
y = y[:-future_days]

#on split notre jeu de données : 

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)

#on instancie le modèle svr (kernel =  type de fonctions pour faire la régréssion,
# C= contrôle de la marge d'erreur, plus il est grand moins la marge d'erreur est permissive
#  gamma= impacte la régularité du modèle)

svr = SVR(kernel = 'rbf', C=1e3, gamma= 0.00001)

#on entraine le modèle

svr.fit(X_train, y_train)

#on affiche le score du modèle
svr_score = svr.score(X_test, y_test)

print(svr_score)

#on instancie les prédiction et on affiche ensuite une comparaison entre les vraies données et les données prédites
svr_pred = svr.predict(X_test)

plt.figure(figsize=(10,10))
plt.plot(svr_pred, label= 'Prédictions', lw = 2, alpha = .7)
plt.plot(y_test, label = 'Vraies valeurs',lw = 2, alpha = .7)
plt.title("Comparaison entre valeurs et prédictions")
plt.ylabel('Prix en $')
plt.xlabel('Date')
plt.legend()
plt.show()
