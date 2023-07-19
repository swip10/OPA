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

df_ticker = postgres.get_table_as_dataframe(ticker)

#On récupère uniquement les datas de la colonne close et timestamp pour notre modèle
df_LM= df_ticker[['timestamp',"close"]]

#On transfert les valeurs de close dans un array numpy
dataset= df_LM.filter(['close']).values

training_data_len= math.ceil(len(dataset)*0.8)

# Scaling de la data

scaler=MinMaxScaler(feature_range=(0,1))

scaled_data= scaler.fit_transform(dataset)

#Création du dataset d'entrainement

train_data = scaled_data[0:training_data_len,:] #On prend les valeurs de 0 à "training data len" et toutes les colonnes correspondantes

# On sépare les data en différents data sets x_train et y_train
x_train=[]
y_train=[]

for i in range(60, len (train_data)):
    x_train.append(train_data[i-60:i,0]) #x_train contient les 60 premières valeurs 
    y_train.append(train_data[i,0])      #y_train contient la 61eme valeur prédite et on incrémente à chaque boucle, 

#On transforme ensuite x_train et y_train en array numpy   
x_train, y_train = np.array(x_train), np.array(y_train)

#on instancie le modèle svr (kernel =  type de fonctions pour faire la régréssion,
# C= contrôle de la marge d'erreur, plus il est grand moins la marge d'erreur est permissive
#  gamma= impacte la régularité du modèle)

svr = SVR(kernel = 'rbf', C=1e3, gamma= 0.00001)

#on entraine le modèle

svr.fit(x_train, y_train)

#Creation de l'array contenant les valeurs pour le dataset test:
test_data=scaled_data[training_data_len-60:, :]
x_test=[]
y_test=dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

#Conversion en un array numpy:
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

#on récupère les prédictions du modele:
predictions=svr.predict(x_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
 #on remet les valeurs au bon format pour les comparer avec celles contenues dans le y_test 

#on affiche le score du modèle
svr_score = svr.score(x_test, y_test)

print(svr_score)

#Visualisation des datas et des prédictions:
train= df_LM[:training_data_len]
valid=df_LM[training_data_len:]
valid['Predictions']=predictions

plt.title('Model LSTM')
plt.xlabel("Date")
plt.ylabel("Close Price USD($)")
plt.plot(train['close'])
plt.plot(valid[['close','Predictions']])
plt.legend(["Train",'Val','Predictions'], loc = 'upper right')
plt.show()