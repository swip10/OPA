from config import config
from src.db.mongodb import MongoOPA
import plotly.graph_objects as go
import numpy as np


def get_wiki_plot(no_failure: bool = True) -> go.Figure:
    try:
        fig = get_wiki_plot_base()
    except Exception as e:
        if no_failure:
            print(e)
            return go.Figure()
        raise e
    else:
        return fig


def get_wiki_plot_base() -> go.Figure:
    client = MongoOPA(
        host=config.mongodb_host,
        port=config.mongodb_port
    )

    client.get_wiki_last_revision()
    df_sentiments = client.get_average_sentiment_over_time()

    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    for step in np.arange(5, 105, 5):
        fig.add_trace(
            go.Scatter(
                visible=False,
                mode='markers',
                line=dict(color="#00CED1", width=6),
                name="rolling average = " + str(step),
                x=df_sentiments["timestamp"],
                y=df_sentiments['average_sentiment'].rolling(window=step).mean()
        ))

    # Make 10th trace visible
    fig.data[10].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Rolling avg window size: " + str(i*5 + 5)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Window: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )
    return fig


if __name__ == "__main__":
    get_wiki_plot().show()
