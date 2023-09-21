import pandas as pd
import requests
from bs4 import BeautifulSoup
from src.db.postgres import Postgres

# Fonction pour mettre à jour et afficher la table 'VOLATILITY'
def run_volatility_script():
    # Faire une requête pour obtenir le contenu de la page web
    url = 'https://fr.tradingview.com/markets/cryptocurrencies/prices-most-volatile/'
    html_content = requests.get(url).text

    # Créer un objet BeautifulSoup pour parser le HTML
    soup = BeautifulSoup(html_content, "lxml")
    # Localiser la table HTML dans la page web (peut nécessiter des ajustements en fonction de la structure du site web)
    table = soup.find_all('table')

    # Trouver toutes les lignes du tableau
    table_rows = table[0].find_all('tr')

    # Initialiser une liste pour stocker les données de chaque ligne
    table_data = []

    # Parcourir chaque ligne
    for row in table_rows:
        # Trouver toutes les cellules de la ligne
        cells = row.find_all('td')

        # Vérifier s'il y a des cellules avant de les analyser
        if cells:
            # Extraire le texte de chaque cellule et le stocker dans une liste
            row_data = []
        
            # Le symbole de la crypto est dans le premier élément, nous l'extrayons différemment
            crypto_symbol = cells[0].find('a').text if cells[0].find('a') else None
            row_data.append(crypto_symbol)

            # Pour le reste des colonnes, nous extrayons simplement le texte
            for cell in cells[1:]:
                row_data.append(cell.text.strip())
            
            # Ajouter les données de la ligne à la liste de données du tableau
            table_data.append(row_data)

    # Convertir la liste de données en DataFrame
    df = pd.DataFrame(table_data)
    postgres_client = Postgres()
    table_names = postgres_client.get_all_table_names()


    # Créez une liste contenant les noms des tables existantes
    existing_tables = [name[0] for name in table_names]

    # Ajoutez une nouvelle colonne "Binance"
    df['binance'] = ''


    # Renommer les colonnes
    df.columns = ['coin', 'rang', 'volatilite_1m', 'prix', 'variation_pourcentage_24h', 
                'capitalisation_boursiere', 'volume_en_usd_24h', 
                'actifs_circulants', 'categorie', 'binance']

    # Définissez une fonction pour vérifier si le nom du coin est dans les tables existantes
    df['binance'] = df['coin'].apply(lambda x: 'OUI' if (x.lower() + 'usdt' in existing_tables) or (x.lower() + 'usd' in existing_tables) else 'NON')

    postgres_client.close()
    return df