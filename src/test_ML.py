from src.db.postgres import Postgres
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
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

# Tracer un graphique avec les données de la colonne "close"
fig = plt.plot(df_LM['timestamp'], df_LM['close'])
plt.title(f"Close price for {ticker} over the last 1k days")
plt.xlabel("Date")
plt.ylabel("Close Price USD ($)")

# Afficher le graphique
plt.show()

#On transfert les valeurs de close dans un array numpy
dataset= df_LM.filter(['close']).values

training_data_len= math.ceil(len(dataset)*0.8)

# print(training_data_len) peut être utilisé pour afficher le nombre de ligne de notre dataset

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

#On reforme les data pour être conforme au format du modele LSTM qui nécessite que nos array soient en 3 dimensions:

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1)) #x_train.shape[0 ou 1] permet d'indiquer les colonnes ou les lignes de l'array

#Construction du modele LSTM:
model= Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1))) #LSTM(nombre de neurones, definition de la sortie (séquence entiere ou derniere sortie), Définition de l'entrée)
model.add(LSTM(50, return_sequences=False) )
model.add(Dense(25)) #ajout de couches de neurones densément connectés
model.add(Dense(1))

#Compilation du modèle
model.compile(optimizer='adam', loss= 'mean_squared_error')

#Entrainement du modèle:

model.fit(x_train,y_train,batch_size=10, epochs=10)  #nombre d'itérations

#Creation de l'array contenant les valeurs pour le dataset test:
test_data=scaled_data[training_data_len-60:, :]
x_test=[]
y_test=dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

#Conversion en un array numpy:
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#on récupère les prédictions du modele:
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions) #on remet les valeurs au bon format pour les comparer avec celles contenues dans le y_test 

#Récupération de la RMSE(root mean squarred error )
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
accuracy = 1 - np.mean(np.abs(predictions - y_test) / y_test)
print("Erreur quadratique moyenne:", rmse)
print("Précision:", accuracy)

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