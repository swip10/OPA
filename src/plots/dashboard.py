# Importation des bibliothèques nécessaires
from dash import Dash, html, dcc, dash_table
import plotly.graph_objects as go
from dash.dependencies import Output, Input
from src.db.postgres import Postgres
from src.plots.wiki import get_wiki_plot
from config.config import CHEMIN_JSON_LOCAL
import plotly.express as px
from tqdm import tqdm
from src.volatility_script import run_volatility_script
from src.plots.wiki import get_wiki_plot_axis


# Initialisation de la connexion à PostgreSQL
postgres = Postgres()

# Obtention de la liste des noms de table
tickers = postgres.get_all_table_names()
dropdown1_options = [{"label": ticker[0], "value": ticker[0]} for ticker in tickers]

# Configuration de l'application Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Configuration de la mise en page de l'application
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Page d'accueil
index_page = html.Div([
    html.H1('OPA dashboard', style={'color': 'aquamarine', 'textAlign': 'center'}),
    
    # Ajoutez cette section pour afficher le nombre de tables
    html.Div(id='table-count', style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    html.Button(dcc.Link('Historical prices from PostGres DB', href='/page-1')),
    html.Br(),
    html.Button(dcc.Link('Find best currency to trade from moving average', href='/page-2')),
    html.Br(),
    html.Button(dcc.Link('Sentiment analysis from MongoDB', href='/page-3')),
    html.Br(),
    html.Button(dcc.Link('Run Volatility Script', href='/page-4')),
    html.Br(),
], style={'alignItems': 'center'})

# Page 1 : Affiche les prix historiques à partir de PostgreSQL
layout_1 = html.Div([
    html.H1('Historical prices from PostGres DB', style={'textAlign': 'center', 'color': 'mediumturquoise'}),

    html.P("Select currency to plot: "),
    html.Div(dcc.Dropdown(id='page-1-dropdown',
                          options=dropdown1_options,
                          value=None
                          )),
    html.Div(dcc.Graph(id='page-1-graph')),

    html.Br(),
    html.Button('Load default example historical csv file', id="load_template", n_clicks=0),
    html.P(id='placeholder'),

    html.Br(),
    html.Button(dcc.Link('Go back to home page', href='/'))
], style={'background': 'beige'})

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
    html.H1('Find best currency to trade from moving average',
            style={'textAlign': 'center', 'color': 'mediumturquoise'}),
    html.Button('Compute', id='loading-input-1', n_clicks=0),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-1")
    ),
    html.Div(id='page2-output'),
    html.Br(),
    html.Button(dcc.Link('Go back to home page', href='/'))
], style={'background': 'beige'})

# Callback pour effectuer le calcul lorsque le bouton "Compute" est cliqué
@app.callback(
    Output(component_id='page2-output', component_property='children'),
    Input(component_id="loading-input-1", component_property="n_clicks"),
)
def input_triggers_spinner(n_clicks):
    if n_clicks > 0:
        message = sma()
        return "", message



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
    df_btc = Postgres().get_table_as_dataframe('btcusdt')
    # Accédez à la plage des axes X
    xaxis_range = get_wiki_plot_axis()
    # Filtrer avec les dates sur l'axe du cours BTC
    df_btc_filtered = df_btc[(df_btc['timestamp'] >= xaxis_range[0]) & (df_btc['timestamp'] <= xaxis_range[1])]
    fig = px.line(df_btc_filtered, x="timestamp", y="close")
    return fig

# Page 3 : Analyse du sentiment à partir de MongoDB
layout_3 = html.Div([
    html.H1('Sentiment analysis from MongoDB', style={'textAlign': 'center', 'color': 'mediumturquoise'}),
    html.Div(dcc.Graph(id='page-3-graph1', figure=get_wiki_plot())),
    html.Br(),
    html.Div(dcc.Graph(id='page-3-graph2', figure=filter_btc_data_by_x_range())),
    html.Br(),
    html.Button(dcc.Link('Go back to home page', href='/'))
], style={'background': 'beige'})


# Page 4 : Affichage du dataframe 'VOLATILITE'
layout_4 = html.Div([
    html.H1('Update and show volatility table',
            style={'textAlign': 'center', 'color': 'mediumturquoise'}),
    html.Button(dcc.Link('Go back to home page', href='/')),
    html.Button('Compute', id='loading-input-4', n_clicks=0),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-4")
    ),
    html.Div(id='page4-output'),
    html.Br(),
    html.Button(dcc.Link('Go back to home page', href='/'))
], style={'background': 'beige'})

# Mise à jour de l'index en fonction de l'URL
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return layout_1
    elif pathname == '/page-2':
        return layout_2
    elif pathname == '/page-3':
        return layout_3
    elif pathname == '/page-4':
        return layout_4
    else:
        # Affichez le nombre de tables sur la page d'accueil
        table_count = get_table_count()
        return html.Div([
            html.H1('OPA dashboard', style={'color': 'aquamarine', 'textAlign': 'center'}),
            html.Div(f'Number of Tables in the Database: {table_count}', style={'textAlign': 'center'}),
            html.Button(dcc.Link('Historical prices from PostGres DB', href='/page-1')),
            html.Br(),
            html.Button(dcc.Link('Find best currency to trade from moving average', href='/page-2')),
            html.Br(),
            html.Button(dcc.Link('Sentiment analysis from MongoDB', href='/page-3')),
            html.Br(),
            html.Button(dcc.Link('Run Volatility Script', href='/page-4')),
            html.Br(),
        ], style={'alignItems': 'center'})

# Callback pour effectuer l'update lorsque le bouton "Compute" est cliqué
@app.callback(
    Output(component_id='page4-output', component_property='children'),
    Input(component_id="loading-input-4", component_property="n_clicks"),
)
def input_triggers_spinner(n_clicks):
    if n_clicks > 0:
        tableau_volat = run_volatility_script()
        return dash_table.DataTable(tableau_volat.to_dict('records'), [{"name": i, "id": i} for i in tableau_volat.columns])
    else:
        return ""  # Afficher le contenu initial vide lorsque le bouton n'est pas encore cliqué


# Fonction pour calculer la meilleure devise à trader
def sma():
    postgres = Postgres()
    table_names = postgres.get_all_table_names()
    table_names = ['btcusdt', 'subbtc', 'ltceur']  # Exemple de devises à analyser
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

    max_wallet = max(wallets, key=wallets.get)
    res = f"Le wallet avec le plus d'argent est {max_wallet} avec {round(wallets[max_wallet], 2)}$ à la fin. \n\n"
    res += f"Nombre de devise(s) pour lesquelles la stratégie n'a pas pu être appliquée : " \
           f"{len(ticker_impossible)}/ {len(table_names)} \n"
    res += f"Exemple de devise(s) pour lesquelles la stratégie n'a pas pu être appliquée: {ticker_impossible[:5]}"
    return res

# Fonction pour obtenir le nombre de tables dans la base de données
def get_table_count():
    postgres = Postgres()
    table_names = postgres.get_all_table_names()
    postgres.close()
    return len(table_names)



# Lancement de l'application Dash
if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")

    