import pandas as pd
import plotly.express as px


df = pd.read_csv("ticker_latest_prices.csv", index_col=0)

fig = px.line(df, x="timestamp", y="bidPrice", color='symbol')
fig.show()
