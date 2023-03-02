import pandas as pd
import plotly.express as px


df = pd.read_csv("ticker_avg_price.csv", index_col=0)

df = df.melt(id_vars=["timestamp"],
             var_name="currency",
             value_name="price")

fig = px.line(df, x="timestamp", y="price", color='currency')
fig.show()
