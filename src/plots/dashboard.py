from dash import Dash
from dash import (
    html,
    dcc,
)
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from dash.dependencies import Output, Input

from src.db.postgres import Postgres
from src.plots.wiki import get_wiki_plot


postgres = Postgres()

tickers = postgres.get_all_table_names()
dropdown1_options = [{"label": ticker[0], "value": ticker[0]} for ticker in tickers]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.H1('OPA dashboard', style={'color': 'aquamarine', 'textAlign': 'center'}),
    html.Button(dcc.Link('Historical prices from PostGres DB', href='/page-1')),
    html.Br(),
    html.Button(dcc.Link('Find best currency to trade from moving average', href='/page-2')),
    html.Br(),
    html.Button(dcc.Link('Sentiment analysis from MongoDB', href='/page-3'))
], style={'alignItems': 'center'})

# Page 1
layout_1 = html.Div([
    html.H1('Historical prices from PostGres DB', style={'textAlign': 'center', 'color': 'mediumturquoise'}),

    html.P("Select currency to plot: "),
    html.Div(dcc.Dropdown(id='page-1-dropdown',
                          options=dropdown1_options,
                          value=None
                          )),
    html.Div(dcc.Graph(id='page-1-graph')),

    html.Br(),
    html.Button(dcc.Link('Go back to home page', href='/'))
], style={'background': 'beige'})


@app.callback(Output(component_id='page-1-graph', component_property='figure'),
              [Input(component_id='page-1-dropdown', component_property='value')])
def update_graph_1(ticker):
    if ticker is None:
        return go.Figure()
    # re-create connection to database mandatory inside callback function
    df = Postgres().get_table_as_dataframe(ticker)
    fig = px.line(df, x="timestamp", y="close")
    return fig


# Page 2
layout_2 = html.Div([
    html.H1('Find best currency to trade from moving average',
            style={'textAlign': 'center', 'color': 'mediumturquoise'}),
    html.Button('Re-compute', id='loading-input-1', n_clicks=0),
    # dcc.Input(id="loading-input-1", value='Input triggers local spinner'),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-1")
    ),
    html.Div(id='page2-output'),
    html.Br(),
    html.Button(dcc.Link('Go back to home page', href='/'))
], style={'background': 'beige'})


@app.callback([Output(component_id="loading-output-1", component_property="children"),
               Output(component_id='page2-output', component_property='children')],
              Input(component_id="loading-input-1", component_property="n_clicks"))
def input_triggers_spinner(_):
    message = sma()
    return "", message


# Page 3
layout_3 = html.Div([
    html.H1('Sentiment analysis from MongoDB', style={'textAlign': 'center', 'color': 'mediumturquoise'}),
    html.Div(dcc.Graph(id='page-3-graph', figure=get_wiki_plot())),

    html.Br(),
    html.Button(dcc.Link('Go back to home page', href='/'))
], style={'background': 'beige'})


# Mise à jour de l'index
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return layout_1
    elif pathname == '/page-2':
        return layout_2
    elif pathname == '/page-3':
        return layout_3
    else:
        return index_page


def sma():
    postgres = Postgres()
    table_names = postgres.get_all_table_names()
    table_names = [name[0] for name in table_names]
    # ici pour l'exemple on choisi que deux ou trois ticker
    # table_names = ['btcusdt', 'subbtc','ltceur']

    # Initialisation d'une liste pour les tickers pour lesquels on ne peux pas appliquer la stratégie
    ticker_impossible = []

    # Initialiser le dictionnaire des wallets pour chaque table
    wallets = {}
    for table in table_names:
        wallets[table] = 1000

    # Parcourir chaque table et calculer les moyennes mobiles
    for table in tqdm(table_names):
        df = postgres.get_table_as_dataframe(table)
        # Skip si pas de colonne 'close' dans les colonnes (autres tables que les tickers)
        if 'close' not in df.columns:
            continue
        # Skip si le dataframe est vide
        if df.empty or len(df) < 200:
            ticker_impossible.append(table)
            wallets.pop(table, None)
            continue

        df['sma20'] = df['close'].rolling(window=20).mean().iloc[49:]
        df['sma50'] = df['close'].rolling(window=50).mean().iloc[49:]

        # Analyser les croisements des moyennes mobiles et acheter ou vendre
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

    # Trouver le wallet avec le plus d'argent à la fin
    max_wallet = max(wallets, key=wallets.get)
    res = f"Le wallet avec le plus d'argent est {max_wallet} avec {round(wallets[max_wallet], 2)}$ à la fin. \n\n"

    # Afficher les autres wallets
    # for wallet in wallets:
    #    if wallet != max_wallet:
    #        print(f"Le wallet {wallet} a {round(wallets[wallet], 2)}$ à la fin.")

    # Affichage des erreurs
    res += f"Nombre ticker(s) pour lesquel la stratégie n'a pas pu être appliquée: " \
           f"{len(ticker_impossible)}/ {len(table_names)} \n"
    res += f"Exemple de ticker(s) pour lesquel la stratégie n'a pas pu être appliquée: {ticker_impossible[:5]}"
    return res


if __name__ == '__main__':
    app.run_server(debug=True, host="127.0.0.1")
