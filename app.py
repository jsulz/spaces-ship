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

"""                                 id author                created_at             last_modified                         subdomain                                               host  likes     sdk      tags  readme_size python_version license duplicated_from models datasets emoji colorFrom colorTo pinned    stage   hardware devMode custom_domains
2            52Hz/CMFNet_deblurring   52Hz 2022-03-02 23:29:35+00:00 2024-05-28 11:25:33+00:00            52hz-cmfnet-deblurring            https://52hz-cmfnet-deblurring.hf.space     17  gradio  [gradio]        924.0           None    None            None   None     None     🍻    indigo  indigo  False  RUNNING  cpu-basic   False             []
3              52Hz/CMFNet_dehazing   52Hz 2022-03-02 23:29:35+00:00 2024-05-28 11:08:25+00:00              52hz-cmfnet-dehazing              https://52hz-cmfnet-dehazing.hf.space      5  gradio  [gradio]        917.0           None    None            None   None     None     ☁      gray    gray  False  RUNNING  cpu-basic   False             []
4            52Hz/CMFNet_deraindrop   52Hz 2022-03-02 23:29:35+00:00 2024-05-30 02:59:24+00:00            52hz-cmfnet-deraindrop            https://52hz-cmfnet-deraindrop.hf.space     10  gradio  [gradio]        920.0           None    None            None   None     None     💦      blue    blue  False  RUNNING  cpu-basic   False             []
5  52Hz/HWMNet_lowlight_enhancement   52Hz 2022-03-02 23:29:35+00:00 2023-05-31 06:37:21+00:00  52hz-hwmnet-lowlight-enhancement  https://52hz-hwmnet-lowlight-enhancement.hf.space      8  gradio  [gradio]       1286.0           None    None            None   None     None     🕶    indigo    None  False  RUNNING  cpu-basic   False             []
7  52Hz/SRMNet_real_world_denoising   52Hz 2022-03-02 23:29:35+00:00 2023-05-31 10:03:04+00:00  52hz-srmnet-real-world-denoising  https://52hz-srmnet-real-world-denoising.hf.space     17  gradio  [gradio]        926.0           None    None            None   None     None     🌪      pink  yellow  False  RUNNING  cpu-basic   False             []"""
