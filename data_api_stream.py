from binance.client import Client
import json
import websockets
import time

# Fonction pour traiter les données reçues par WebSocket
def on_message(ws, message):
    data = json.loads(message)
    # Vérifier si c'est un message de prix ticker
    if 'e' in data and data['e'] == 'ticker':
        symbol = data['s']  # Récupérer le symbole
        price = float(data['c'])  # Récupérer le prix en float
        # Ajouter les données à un dictionnaire
        ticker_data[symbol] = price
        # Enregistrer les données dans un fichier JSON toutes les 5 secondes
        if time.time() - last_save_time > 5:
            with open('ticker_data.json', 'w') as f:
                json.dump(ticker_data, f)
            last_save_time = time.time()

# Fonction pour gérer les erreurs WebSocket
def on_error(ws, error):
    print(error)

# Fonction pour gérer la fermeture de la connexion WebSocket
def on_close(ws):
    print("WebSocket closed")

# Fonction pour ouvrir la connexion WebSocket
def on_open(ws):
    # Envoyer une demande d'abonnement pour les prix de tous les tickers
    ws.send(json.dumps({
        "method": "SUBSCRIBE",
        "params": ["!ticker@arr"],
        "id": 1
    }))

if __name__ == "__main__":
    # Initialiser le dictionnaire pour stocker les données ticker
    ticker_data = {}
    # Initialiser le temps de dernière sauvegarde
    last_save_time = time.time()
    # Ouvrir une connexion WebSocket
    ws = websockets.WebSocketApp("wss://stream.binance.com:9443/ws", on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()  # démarrer la boucle infinie pour recevoir des données en continu
