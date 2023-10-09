# Importation des bibliothèques nécessaires
# Bibliothèques
import json
import joblib
import numpy as np
import pandas as pd
import dash_canvas as dc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.model.time_serie_base_line_model import TimeSerieBaseLineModel
from config import config
from src.db.postgres import Postgres
import dash
import dash_bootstrap_components as dbc
from binance.client import Client
from dash import Dash, html, dcc, dash_table
import plotly.graph_objects as go
from dash.dependencies import Output, Input, State
from src.db.postgres import Postgres
from src.plots.wiki import get_wiki_plot
from config.config import CHEMIN_JSON_LOCAL
import requests
import plotly.express as px
from tqdm import tqdm
from src.volatility_script import run_volatility_script
from src.plots.wiki import get_wiki_plot_axis
from src.postgreSQL_stream_script import launch_stream
import time
import atexit
from typing import Dict, List, Union

# Votre code commence ici


app = dash.Dash(external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)

# Organisation de la sidebar de la page d'accueil
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}
# Organisation générale du contenu
CONTENT_STYLE = {
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Barre de navigation avec brand à gauche et barre de recherche à droite
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/", className="nav-link align-items-center text-center")),
        dbc.NavItem(dbc.NavLink("Historical prices", href="/page-1", className="nav-link align-items-center text-center")),
        dbc.NavItem(dbc.NavLink("Best currency to trade", href="/page-2", className="nav-link align-items-center text-center")),
        dbc.NavItem(dbc.NavLink("Sentiment analysis", href="/page-3", className="nav-link align-items-center text-center")),
        dbc.NavItem(dbc.NavLink("Crypto Volatility", href="/page-4", className="nav-link align-items-center text-center")),
        dbc.NavItem(dbc.NavLink("Stock market prediction", href="/page-5", className="nav-link align-items-center text-center")),
        dbc.NavItem(dbc.NavLink("Machine Learning", href="/page-6", className="nav-link align-items-center text-center")),
        dbc.NavItem(dbc.NavLink("Streaming", href="/page-7", className="nav-link align-items-center text-center")),
        dbc.Form(
            dbc.Input(type="search", placeholder="Search", className="form-control"),
            className="d-flex",  
        ),
        dbc.Button("Search", color="secondary", className="ms-2"),
    ],
    brand="OPA dashboard",
    brand_href="#",
    color="primary",
    dark=True,
    brand_style={"margin-left": "5px"},
    fluid=True,
    style={"display": "flex", "justify-content": 'space-between', "align-items": "center"}  
)



# Initialisation de la connexion à PostgreSQL
postgres = Postgres()

# Initialisation du fichier pour le streaming
try:
    with open('start_streaming.txt', 'w') as f:
        f.write("stop\ncurrency\ntimestamp")
except Exception as e:
    print(f"Erreur lors de la création du fichier: {e}")

# Obtention de la liste des noms de table
tickers = postgres.get_all_table_names()
dropdown1_options = [{"label": ticker[0], "value": ticker[0]} for ticker in tickers]


content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"),  content])


# Organisation de la page d'accueil
index_page = html.Div(
    [
        html.H2("OPA dashboard", className="display-4", style={"width": "90%", "text-align": "center", "font-size": "40px","font-weight": "bold"}),
        html.Hr(),
        html.P("A simple App to help trading cryptocurrencies", className="lead",style={"width": "90%", "text-align": "center"}),
        html.Div(id='table-count', style={'textAlign': 'center', 'marginBottom': '20px'}),
        dbc.Button("Historical prices from PostGres DB", href="/page-1", color="primary", className="mb-2", style={"width": "90%", "text-align": "center"}),
        dbc.Button("Find best currency to trade from moving average", href="/page-2", color="primary", className="mb-2", style={"width": "90%", "text-align": "center"}),
        dbc.Button("Sentiment analysis from MongoDB", href="/page-3", color="primary", className="mb-2", style={"width": "90%", "text-align": "center"}),
        dbc.Button("Run Volatility Script", href="/page-4", color="primary", className="mb-2", style={"width": "90%", "text-align": "center"}),
        dbc.Button("Predict next stock market prices", href="/page-5", color="primary", className="mb-2", style={"width": "90%", "text-align": "center"}),
        dbc.Button("Machine Learning", href="/page-6", color="primary", className="mb-2", style={"width": "90%", "text-align": "center"}),
        dbc.Button("Streaming", href="/page-7", color="primary", className="mb-2", style={"width": "90%", "text-align": "center"}),
    ],
    style=SIDEBAR_STYLE
)



# Page 1 : Affiche les prix historiques à partir de PostgreSQL
layout_1 = html.Div([
    navbar,

    html.H1('Historical prices from PostGres DB', style={'textAlign': 'center'}),

    html.H5("Select currency to plot: "),
    html.Div(dcc.Dropdown(id='page-1-dropdown',
                          options=dropdown1_options,
                          value=None
                          )),
    html.Div(dcc.Graph(id='page-1-graph')),

    html.Br(),
    dbc.Button('Load default historical csv file', id="load_template", color="primary", className="mb-2", style={"width": "15%", "text-align": "center"}, n_clicks=0),
    html.P(id='placeholder'),

    html.Br(),
    
])


# Callback pour mettre à jour le graphique en fonction de la sélection de la devise
@app.callback(Output(component_id='page-1-graph', component_property='figure'),
              [Input(component_id='page-1-dropdown', component_property='value')])
def update_graph_1(ticker):
    if ticker is None:
        return go.Figure()
    df = Postgres().get_table_as_dataframe(ticker)
    fig = px.line(df, x="timestamp", y="close")
    return fig


# Callback pour charger un fichier CSV par défaut
@app.callback(
    Output("page-1-dropdown", "options"),
    Input('load_template', 'n_clicks'),
)
def load_default_csv_file(n_clicks):
    print("nombre de clicks", n_clicks)
    global dropdown1_options
    if n_clicks == 0 or len(dropdown1_options) != 0:
        return dropdown1_options
    postgres_client = Postgres()
    postgres_client.initialize_with_historical_json(CHEMIN_JSON_LOCAL, reset=True)
    tickers = postgres_client.get_all_table_names()
    postgres_client.close()
    new_list = [{"label": ticker[0], "value": ticker[0]} for ticker in tickers]
    dropdown1_options = new_list
    return dropdown1_options


# Page 2 : Calcule la meilleure devise à trader à partir de la moyenne mobile
layout_2 = html.Div([
    navbar,

    html.H1('Find best currency to trade from moving average',
            style={'textAlign': 'center'}),
    dbc.Button('Compute', id='loading-input-1', n_clicks=0),
    html.Br(),
    dbc.Button('Compute SMA_all', id='compute-sma_all', n_clicks=0),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-1")
    ),
    html.Div(id='page2-output'),
    dash_table.DataTable(id='table-output', style_table={'overflowX': 'auto'}),
    html.Br(),
    
])


# Callback pour effectuer le calcul lorsque le bouton "Compute" est cliqué
@app.callback(
    [Output(component_id='page2-output', component_property='children'),
     Output('table-output', 'data'),
     Output('table-output', 'columns')],
    [Input(component_id="loading-input-1", component_property="n_clicks"),
     Input('compute-sma_all', 'n_clicks')],
)
def input_triggers_spinner(n_clicks, n_clicks_all):
    ctx = dash.callback_context
    result_df = pd.DataFrame()  # Initialiser avec une valeur par défaut
    message = ""  # Initialiser avec une valeur par défaut

    if not ctx.triggered_id:
        button_id = 'loading-input-1'
    else:
        button_id = ctx.triggered_id.split('.')[0]

    if button_id == 'loading-input-1' and n_clicks > 0:
        message, result_df = sma2()
    elif button_id == 'compute-sma_all' and n_clicks_all > 0:
        message, result_df = sma()

    if result_df.empty:
        data, columns = [], []
    else:
        data = result_df.to_dict('records')
        columns = [{'name': col, 'id': col} for col in result_df.columns]

    return message, data, columns


def get_btc_plot(no_failure: bool = True) -> go.Figure:
    try:
        df = Postgres().get_table_as_dataframe('btcusdt')
        fig = px.line(df, x="timestamp", y="close")
        
    except Exception as e:
        if no_failure:
            print(e)
            return go.Figure()
        raise e
    else:
        return fig


# Fonction pour filtrer les données du BTC en fonction de l'axe des x du graphique 1
def filter_btc_data_by_x_range(no_failure: bool = True) -> go.Figure:
    try:
        df_btc = Postgres().get_table_as_dataframe('ETHBTC')
        # Accédez à la plage des axes X
        xaxis_range = get_wiki_plot_axis()
        # Filtrer avec les dates sur l'axe du cours BTC
        df_btc_filtered = df_btc[(df_btc['timestamp'] >= xaxis_range[0]) & (df_btc['timestamp'] <= xaxis_range[1])]
        fig = px.line(df_btc_filtered, x="timestamp", y="close")
    except Exception as e:
        if no_failure is False:
            raise e
        fig = go.Figure()
    return fig


# Page 3 : Analyse du sentiment à partir de MongoDB
layout_3 = html.Div([
    navbar,

    html.H1('Sentiment analysis from MongoDB', style={'textAlign': 'center'}),
    html.Div(dcc.Graph(id='page-3-graph1', figure=get_wiki_plot())),
    html.Br(),
    html.Div(dcc.Graph(id='page-3-graph2', figure=filter_btc_data_by_x_range())),
    html.Br(),
    ])

# Page 4 : Affichage du dataframe 'VOLATILITE'
layout_4 = html.Div([
    navbar,

    html.H1('Update and show volatility table',
            style={'textAlign': 'center'}),
    dbc.Button('Compute', id='loading-input-4', n_clicks=0),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-4")
    ),
    html.Div(id='page4-output'),
    html.Br(),
    ])

# Page 5 : Prédictions
layout_5 = html.Div([
    navbar,
    
    html.H1('Predict next stock market prices',
            style={'textAlign': 'center'}),
    html.H5('It will download the 59 last close market price for ETHBTC spaced by 8 hours and '
            'run a model prediction for future values'),

    # Ajout d'un élément pour l'option de test (True ou False)
    dbc.Switch(
        id='testing-option',
        value=True,
        label= 'Testing',
        className='mt-4'),

    html.Div([
    dcc.Input(
        id='prediction-count',
        type='number',
        value=5,  # Par défaut, 5 prédictions
        style={'margin-right': '10px'}  # Ajoutez une marge à droite pour espacer les éléments
    ),
    dbc.Button('Compute Predictions', id='compute-predictions', n_clicks=0),
    ], style={'display': 'flex', 'align-items': 'center'}),

    html.Div(dcc.Graph(id='page-5-graph1', figure=go.Figure())),
    html.Br(),
    ])

# Page 6
layout_6 = html.Div([
    navbar,
    html.H1('Machine Learning', style={'textAlign': 'center'}),
    html.H5("Select currency to use for training: "),
    html.Div(dcc.Dropdown(id='page-6-dropdown',
                          options=dropdown1_options,
                          value=None
                          )),
    dbc.Switch(id='save-model', value=False, label='Save model', className='mt-4'),
    html.H5("Number of Epochs: "),
    dbc.Input(id='num-epochs-input', type='number', value=1, style={"width": "15%", 'margin-bottom':'10px'}),
    dbc.Button("Run Script", id="run-script-button", n_clicks=0),
    html.Div(dcc.Graph(id='page-6-graph1', figure=go.Figure())),
    html.Br()
])


#définition de la fonction permettant d'entrainer le modèle et d'afficher le graphique:

def train_and_generate_graph(selected_ticker, save_model,num_epochs):
    
    df = Postgres().get_table_as_dataframe(selected_ticker)

    df.sort_values(by='timestamp', ascending=True, inplace=True)
    # Sélectionnez uniquement les colonnes 'close' et 'volume'
    df = df[['close', 'volume']]

    SAVE_MODEL = save_model == True

    scaler = MinMaxScaler()
    close_price = df.close.values.reshape(-1, 1)
    scaled_close = scaler.fit_transform(close_price)
    scaler_volume = MinMaxScaler()
    scaled_volume = scaler_volume.fit_transform(df.volume.values.reshape(-1, 1))

    if SAVE_MODEL:
        joblib.dump(scaler, 'scaler_close_price.save')
        joblib.dump(scaler_volume, 'scaler_volume.save')

    sequence_len = 60

    scaled_data = np.concatenate((scaled_close, scaled_volume), axis=1)

    def split_into_sequences(data, seq_len):
        n_seq = len(data) - seq_len + 1
        return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

    def get_train_test_sets(data, seq_len, train_frac):
        sequences = split_into_sequences(data, seq_len)
        n_train = int(sequences.shape[0] * train_frac)
        x_train = sequences[:n_train, :-1, :]
        y_train = sequences[:n_train, -1, :]
        x_test = sequences[n_train:, :-1, :]
        y_test = sequences[n_train:, -1, :]
        return x_train, y_train, x_test, y_test

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=False)
    x_train, y_train, x_test, y_test = get_train_test_sets(scaled_data, sequence_len, train_frac=0.9)

    batch_size = 64
    model = TimeSerieBaseLineModel(
        seq_len=sequence_len,
        input_shape=x_train.shape[-1],
        output_shape=y_train.shape[-1],
        dropout=0.05
    )
    model.currency = selected_ticker

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
    )

    model.build((None, sequence_len, x_train.shape[-1]))
    
    model.fit(
        x_train,
        y_train,
        epochs= num_epochs,
        batch_size=batch_size,
        shuffle=False,
        validation_data=(x_test, y_test)
    )

    if SAVE_MODEL:
        model.save("keras_next")
    
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    y_train = np.expand_dims(y_train[:, 0], axis=-1)
    y_test = np.expand_dims(y_test[:, 0], axis=-1)
    y_pred = np.expand_dims(y_pred[:, 0], axis=-1)
    y_pred_train = np.expand_dims(y_pred_train[:, 0], axis=-1)

    y_test_orig = scaler.inverse_transform(y_test)
    y_pred_orig = scaler.inverse_transform(y_pred)
    y_pred_train_orig = scaler.inverse_transform(y_pred_train)

    y_train2= np.squeeze(y_train)
    y_test2= np.squeeze(y_test)
    y_pred2=np.squeeze(y_pred)
    y_pred_train2=np.squeeze(y_pred_train)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, len(y_train)), y=y_train2, mode='lines', name='Historical Price', line=dict(color='brown')))
    fig.add_trace(go.Scatter(x=np.arange(len(y_train), len(y_train) + len(y_test_orig)), y=y_test2, mode='lines', name='Actual Price', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=np.arange(len(y_train), len(y_train) + len(y_pred_orig)), y=y_pred2, mode='lines', name='Predicted Price', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=np.arange(0, len(y_train)), y=y_pred_train2, mode='lines', name='Predicted Price train', line=dict(color='blue')))

    fig.update_layout(
        title=f'{selected_ticker} 8hours Prices',
        xaxis_title='8hours space',
        yaxis_title='Price ($)',
            )

    return fig

# Callback pour l'entraînement du modèle en fonction de la devise sélectionnée
@app.callback(
    Output('page-6-graph1', 'figure'),
    Input('run-script-button', 'n_clicks'),
    State('page-6-dropdown', 'value'),
    State('save-model', 'on'),
    State('num-epochs-input', 'value')
)
def update_graph(n_clicks, selected_ticker, save_model, num_epochs):
    if n_clicks > 0:
        # Appelez votre fonction train_and_generate_graph avec num_epochs
        fig = train_and_generate_graph(selected_ticker, save_model, num_epochs)
        return fig
    else:
        return go.Figure()

#page 7

layout_7 = html.Div([
    navbar,
    
    html.H1('Streaming Page', style={'textAlign': 'center'}),
    
    dcc.Graph(id='live-update-graph'),  # Graphique pour afficher les données
    html.P("Select currency to plot: "),
    html.Div(dcc.Dropdown(id='page-1-dropdown',
                          options=dropdown1_options,
                          value=None
                          )),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # en millisecondes, donc 1*1000 = 1 seconde
        n_intervals=0
    )
])

@app.callback(
    Output('live-update-graph', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('page-1-dropdown', 'value')]
)
def update_graph_live(n, selected_currency):
    # Si aucune monnaie n'est sélectionnée, vous pouvez soit ne rien afficher, soit afficher une monnaie par défaut
    if not selected_currency:
        try:
            with open('start_streaming.txt', 'w') as f:
                f.write("stop\ncurrency\ntimestamp")
        except Exception as e:
            print(f"Erreur lors de la modification du fichier: {e}")
        print('Rien')
        return dash.no_update
    
    # Si une nouvelle monnaie est sélectionnée
    if selected_currency:
    # Si un streamer existe déjà, arrêtez-le et remplacez par le prochain
        print('lancement stream')
        if should_write_to_file('start', selected_currency):
            try:
                with open('start_streaming.txt', 'w') as f:
                    f.write(f'start\n{selected_currency}\ntimestamp')
            except Exception as e:
                print(f"Erreur lors de la modification du fichier: {e}")


    # Utilisez la monnaie sélectionnée pour récupérer les données
    try:
        df = Postgres().get_table_as_dataframe_stream(selected_currency)
        # Mise à jour du graphique avec les nouvelles données
        fig = px.line(df, x='timestamp', y='close', title=f'Live streaming data for {selected_currency.upper()}')
        return fig
    except Exception as e:
        # Gérer l'exception et peut-être retourner un message d'erreur ou un graphique vide
        print(e)
        return dash.no_update


# Mise à jour de l'index en fonction de l'URL

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    # Si on navigue hors de la page de streaming
    if pathname != '/page-7':
        stop_streaming()
    # Si l'URL est la page d'accueil, affichez la sidebar
    if pathname == "/":
        return html.Div([
    index_page,
    html.Div([
        html.H1("Welcome to our Tradingbot App !", style={"text-align": "center", "font-size": "72px", "margin-top": "35vh", "margin-left":'25vh'}), 
        html.Div([html.H4(f'Number of Tables in the Database: {table_count}', style={'textAlign': 'center'})],
                 style={"position": "fixed", "bottom": "10px", "left": "0", "right": "0", "background": "#f8f9fa"}),
    ], style={"display": "flex", "flex-direction": "column", "justify-content": "center", "align-items": "center"}),
])
    elif pathname == '/page-1':
        return layout_1
    elif pathname == '/page-2':
        return layout_2
    elif pathname == '/page-3':
        return layout_3
    elif pathname == '/page-4':
        return layout_4
    elif pathname == '/page-5':
        return layout_5
    elif pathname == '/page-6':
        return layout_6
    elif pathname == '/page-7':
        return layout_7
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


# Callback pour effectuer l'update lorsque le bouton "Compute Predictions" est cliqué
@app.callback(
    Output(component_id='page-5-graph1', component_property='figure'),
    Input(component_id="compute-predictions", component_property="n_clicks"),
    [State("testing-option", "value"), State("prediction-count", "value")],
)
def update_prediction_stock_market_price_figure(n_clicks: int, testing_option: list, prediction_count: int) -> go.Figure:
    """
    triggered after clicking 'compute' button
    :param n_clicks: (int) number of clicks
    :param testing_option: (list) selected testing options
    :param prediction_count: (int) number of predictions
    :return: (go.Figure)
    """
    # On récupère les options sélectionnées pour le test (True ou False)
    testing = testing_option == True

    # On récupère le nombre de prédictions souhaitées
    NUMBER_OF_PREDICTIONS = prediction_count

    print(n_clicks)
    if n_clicks > 0:
        
        ticker = "ETHBTC"  # model has been trained with this ticker

        client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
        client.ping()

        nb_samples = 8 * 59  # 59 values spaced by 8 jours
        nb_samples += NUMBER_OF_PREDICTIONS * 8 if testing else 0  # Utilisez la variable de test
        klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_8HOUR, f"{nb_samples} hours ago")
        data = pd.DataFrame(
            klines,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ]
        )
        data = data[["close", "volume"]]
        data = data.astype(float)
        json_data = {"price": list(data["close"][:59]), "volume": list(data["volume"][:59]), "currency": ticker}

        url = f"http://{config.fastapi_host}:{config.fastapi_port}/prediction"

        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }

        params = {
            'next_hours': str(NUMBER_OF_PREDICTIONS * 8),
        }

        r = requests.post(url, params=params, headers=headers, json=json_data)

        response = r.json()
        x_true = np.arange(len(data["close"]))
        y_pred = response["prices"]
        x_pred = np.arange(59, 59 + NUMBER_OF_PREDICTIONS)
        fig = go.Figure(
            go.Scatter(x=x_true, y=data["close"], mode='lines+markers')
        )
        fig.add_trace(
            go.Scatter(x=x_pred, y=y_pred, mode='markers')
        )
        return fig
    else:
        return go.Figure()



@app.callback(
    Output(component_id='page4-output', component_property='children'),
    Input(component_id="loading-input-4", component_property="n_clicks"),
)
def input_triggers_spinner(n_clicks: int) -> dash_table.DataTable:
    """
    triggered after clicking 'compute' button
    :param n_clicks: (int) number of clicks
    :return: (DataTable)
    """
    if n_clicks > 0:
        tableau_volat = run_volatility_script()
        return dash_table.DataTable(tableau_volat.to_dict('records'), [{"name": i, "id": i} for i in tableau_volat.columns])
    else:
        return dash_table.DataTable()

def sma2() -> str:
    """
    Fonction pour calculer la meilleure devise à trader
    :return: message: (str)
    """
    postgres = Postgres()
    table_names = postgres.get_all_table_names()

    table_names = [name[0] for name in table_names if name[0].lower().endswith(('usd', 'usdt'))]
    table_names = table_names[0:10]
    ticker_impossible = []  # Liste pour les devises pour lesquelles la stratégie ne peut pas être appliquée
    wallets = {}  # Dictionnaire pour stocker les résultats

    for table in table_names:
        wallets[table] = 1000

    for table in tqdm(table_names):
        df = postgres.get_table_as_dataframe(table)
        if 'close' not in df.columns:
            continue
        if df.empty or len(df) < 200:
            ticker_impossible.append(table)
            wallets.pop(table, None)
            continue

        df['sma20'] = df['close'].rolling(window=20).mean().iloc[49:]
        df['sma50'] = df['close'].rolling(window=50).mean().iloc[49:]

        in_position = False
        for i in range(len(df)):
            if df['sma20'][i] > df['sma50'][i] and not in_position:
                buy_price = df['close'][i]
                in_position = True
            elif df['sma20'][i] < df['sma50'][i] and in_position:
                sell_price = df['close'][i]
                profit = (sell_price - buy_price) * (wallets[table] / buy_price)
                wallets[table] += profit
                in_position = False
    # Construire result_df comme un DataFrame
    result_df = pd.DataFrame({
        'Table': list(wallets.keys()),
        'End Wallet': [round(wallet, 2) for wallet in wallets.values()]
    })

    max_wallet = max(wallets, key=wallets.get)
    res = f"Le wallet avec le plus d'argent est {max_wallet} avec {round(wallets[max_wallet], 2)}$ à la fin. \n\n"
    res += f"Nombre de devise(s) pour lesquelles la stratégie n'a pas pu être appliquée : " \
           f"{len(ticker_impossible)}/ {len(table_names)} \n"
    res += f"Exemple de devise(s) pour lesquelles la stratégie n'a pas pu être appliquée: {ticker_impossible[:5]}"
    return res, result_df

def sma() -> str:
    postgres=Postgres()
    # On récupère le nom des différentes tables de la base de donnée, ainsi que le df pour le ticker choisi (nom, ou position dans la liste obtenue)
    # table_names=postgres.get_all_table_names()
    # ici pour l'exemple on choisi que deux ticker ticker
    table_names = postgres.get_all_table_names()
    table_names = [name[0] for name in table_names if name[0].lower().endswith(('usd', 'usdt'))]
    table_names = table_names[0:3]


    ticker_impossible = []  # Liste pour les devises pour lesquelles la stratégie ne peut pas être appliquée
    # Initialiser le dictionnaire de résultats
    results = {}
    for table in table_names:
        results[table] = []

    # Parcourir chaque table et tester chaque combinaison de SMA
    for table in table_names:
        df = postgres.get_table_as_dataframe(table)
        if 'close' not in df.columns:
            continue
        if df.empty or len(df) < 300:
            ticker_impossible.append(table)
            wallets.pop(table, None)
            continue

        # Initialiser le dictionnaire des wallets pour chaque combinaison de SMA
        wallets = {}
        for i in range(20, 30, 2):
            for j in range(50, 201, 10):
                if i < j:
                    wallets[(i, j)] = 1000

        # Parcourir chaque combinaison de SMA et calculer les résultats
        for i in range(20, 30, 4):
            for j in range(50, 191, 20):
                if i < j:
                    df['sma'+str(i)] = df['close'].rolling(window=i).mean()
                    df['sma'+str(j)] = df['close'].rolling(window=j).mean() 

                    # Analyser les croisements des moyennes mobiles et acheter ou vendre
                    in_position = False
                    for k in range(len(df)):
                        if df['sma'+str(i)][k] > df['sma'+str(j)][k] and not in_position:
                            buy_price = df['close'][k]
                            in_position = True
                            wallet = wallets[(i, j)]
                        elif df['sma'+str(i)][k] < df['sma'+str(j)][k] and in_position:
                            sell_price = df['close'][k]
                            profit = (sell_price - buy_price) * (wallet / buy_price)
                            wallet += profit
                            wallets[(i, j)] = wallet
                            in_position = False

                    # Ajouter les résultats pour cette combinaison de SMA
                    results[table].append({
                        'sma1': i,
                        'sma2': j,
                        'wallet': max(wallets.values())
                    })

    # Construire le DataFrame de résultats
    result_df = build_results_dataframe(results)
    # Afficher les résultats
    # Trouver le meilleur résultat pour chaque table
    for table in table_names:
        best_result = max(results[table], key=lambda x: x['wallet'])
        res=f"Table {table}: SMA1={best_result['sma1']}, SMA2={best_result['sma2']}, "
        res +=f"Wallet={round(best_result['wallet'], 2)}"
        return res, result_df

def build_results_dataframe(results: Dict[str, List[Dict[str, Union[int, float]]]]) -> pd.DataFrame:
    data = {'Table': [], 'Best_SMA1': [], 'Best_SMA2': [], 'Best_Wallet': []}
    for table, result_list in results.items():
        if not result_list:  # Vérification si la liste est vide
            continue
        best_result = max(result_list, key=lambda x: x['wallet'])
        data['Table'].append(table)
        data['Best_SMA1'].append(best_result['sma1'])
        data['Best_SMA2'].append(best_result['sma2'])
        data['Best_Wallet'].append(round(best_result['wallet'], 2))

    result_df = pd.DataFrame(data)
    return result_df


def get_table_count() -> int:
    """ Fonction pour obtenir le nombre de tables dans la base de données
    :return: count(int): number of tables
    """
    postgres = Postgres()
    table_names = postgres.get_all_table_names()
    postgres.close()
    return len(table_names)

table_count= get_table_count()

def should_write_to_file(command, currency):
    try:
        with open('start_streaming.txt', 'r') as f:
            lines = f.readlines()
        
        # Si le fichier contient au moins 2 lignes et qu'elles correspondent à ce qui est attendu
        if len(lines) >= 2 and lines[0].strip() == command and lines[1].strip() == currency:
            return False
        return True
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")
        return True  # Si nous rencontrons une erreur, nous considérons que nous devons écrire dans le fichier

def stop_streaming():
    """Arrêtez le streaming en mettant à jour le fichier."""
    try:
        with open('start_streaming.txt', 'w') as f:
            f.write("stop\ncurrency\ntimestamp")
    except Exception as e:
        print(f"Erreur lors de la modification du fichier: {e}")

# Ajoutez ceci pour vous assurer que stop_streaming est appelé à la sortie.
atexit.register(stop_streaming)

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")

    