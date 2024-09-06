import gradio as gr
import pandas as pd

# Load the spaces.parquet file as a dataframe
df = pd.read_parquet("spaces.parquet")

"""
Todos:
    Create tabbed interface for filtering and graphs
    plotly graph showing the growth of spaces over time
    plotly graph showing the breakdown of spaces by sdk
    plotly graph of colors
    plotly graph of emojis
    Plotly graph of hardware
    Investigate README lengths
    bar chart of the number of spaces per author
"""


def filtered_df(emoji, likes):
    _df = df
    # if emoji is not none, filter the dataframe with it
    if emoji:
        _df = _df[_df["emoji"].isin(emoji)]
    # if likes is not none, filter the dataframe with it
    if likes:
        _df = _df[_df["likes"] >= likes]
    return _df


with gr.Blocks() as demo:
    df = df[df["stage"] == "RUNNING"]
    emoji = gr.Dropdown(
        df["emoji"].unique().tolist(), label="Search by Emoji 🤗", multiselect=True
    )  # Dropdown to select the emoji
    likes = gr.Slider(
        minimum=df["likes"].min(),
        maximum=df["likes"].max(),
        step=1,
        label="Filter by Likes",
    )  # Slider to filter by likes
    hardware = gr.Dropdown(
        df["hardware"].unique().tolist(), label="Search by Hardware", multiselect=True
    )
    devMode = gr.Checkbox(value=False, label="DevMode Enabled")
    clear = gr.ClearButton(components=[emoji])

    df = pd.DataFrame(df[["emoji", "host", "likes", "hardware"]])
    df["host"] = df["host"].apply(lambda x: f"<a href={x}>{x}</a>")
    gr.DataFrame(filtered_df, inputs=[emoji, likes], datatype=["str", "html"])


print(df.head())
demo.launch()
