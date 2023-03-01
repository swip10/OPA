import asyncio
import websockets
import json
import time

async def subscribe(websocket):
    # Abonnement au ticker de BTCUSDT
    await websocket.send('{"method": "SUBSCRIBE", "params": ["btcusdt@ticker"], "id": 1}')

async def receive(websocket, data_list):
    async for message in websocket:
        data = json.loads(message)
        print(data)
        data_list.append(data)
        break

async def main():
    async with websockets.connect("wss://stream.binance.com:9443/ws") as websocket:
        await subscribe(websocket)

        # Mettre en place une boucle d'attente de 10 secondes
        # et d'arrêt après 1 minute (6 boucles)
        data_list = []
        for i in range(6):
            await receive(websocket, data_list)
            time.sleep(10)

        # Écrire les données dans un fichier JSON
        with open("BTCUSDT_data_stream.json", "w") as json_file:
            json.dump(data_list, json_file)

if __name__ == "__main__":
    asyncio.run(main())
