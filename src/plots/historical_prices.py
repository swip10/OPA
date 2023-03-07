import json
import pandas as pd
import plotly.express as px

with open("../ticker_data_hist.json", "r") as json_file:
    hist_data = json.load(json_file)

list_df = []
for key in hist_data:
    sub_df = pd.DataFrame(hist_data[key])
    sub_df["symbol"] = key
    list_df.append(sub_df)
df = pd.concat(list_df, ignore_index=True)
col_float = [c for c in df if "timestamp" != c and "symbol" != c]
df[col_float] = df[col_float].astype(float)

fig = px.line(df, x="timestamp", y="close", color='symbol')
fig.show()

df = pd.read_csv("ticker_avg_price.csv", index_col=0)
df = df.astype(float)

df = df.melt(id_vars=["timestamp"],
             var_name="currency",
             value_name="price")

fig = px.line(df, x="timestamp", y="price", color='currency')
fig.show()
