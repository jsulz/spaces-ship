import gradio as gr
import pandas as pd
import numpy as np

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
    Is there a correlation between pinning a space and the number of likes?
    Is a correlation between the emoji and the number of likes?
    distribution of python versions
    what models are most used
    what organizations are most popular in terms of their models and datasets being used
    most duplicated spaces

        "id",
        "author",
        "created_at",
        "last_modified",
        "subdomain",
        "host",
        "likes",
        "sdk",
        "tags",
        "readme_size",
        "python_version",
        "license",
        "duplicated_from",
        "models",
        "datasets",
        "emoji",
        "colorFrom",
        "colorTo",
        "pinned",
        "stage",
        "hardware",
        "devMode",
        "custom_domains",    
"""


def filtered_df(emoji, likes, author, hardware, tags, models, datasets):
    _df = df
    # if emoji is not none, filter the dataframe with it
    if emoji:
        _df = _df[_df["emoji"].isin(emoji)]
    # if likes is not none, filter the dataframe with it
    if likes:
        _df = _df[_df["likes"] >= likes]
    if author:
        _df = _df[_df["author"].isin(author)]
    if hardware:
        _df = _df[_df["hardware"].isin(hardware)]
    # check to see if the array of sdk_tags contains any of the selected tags
    if tags:
        _df = _df[_df["sdk_tags"].apply(lambda x: any(tag in x for tag in tags))]
    if models:
        _df = _df[
            _df["models"].apply(
                lambda x: (
                    any(model in x for model in models) if x is not None else False
                )
            )
        ]
    if datasets:
        _df = _df[
            _df["datasets"].apply(
                lambda x: (
                    any(dataset in x for dataset in datasets)
                    if x is not None
                    else False
                )
            )
        ]
    return _df


with gr.Blocks() as demo:
    df = df[df["stage"] == "RUNNING"]
    # combine the sdk and tags columns, one of which is a string and the other is an array of strings
    # first convert the sdk column to an array of strings
    df["sdk"] = df["sdk"].apply(lambda x: np.array([x]))
    # then combine the sdk and tags columns so that their elements are together
    df["sdk_tags"] = df[["sdk", "tags"]].apply(
        lambda x: np.concatenate((x[0], x[1])), axis=1
    )

    # where the custom_domains column is not null, use that as the url, otherwise, use the host column
    df["url"] = np.where(
        df["custom_domains"].isnull(),
        df["id"],
        df["custom_domains"],
    )
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
    author = gr.Dropdown(
        df["author"].unique().tolist(), label="Search by Author", multiselect=True
    )
    # get the list of unique strings in the sdk_tags column
    sdk_tags = np.unique(np.concatenate(df["sdk_tags"].values))
    # create a dropdown for the sdk_tags
    sdk_tags = gr.Dropdown(
        sdk_tags.tolist(), label="Filter by SDK/Tags", multiselect=True
    )
    # create a gradio checkbox group for hardware
    hardware = gr.CheckboxGroup(
        df["hardware"].unique().tolist(), label="Filter by Hardware"
    )

    space_license = gr.CheckboxGroup(
        df["license"].unique().tolist(), label="Filter by license"
    )

    # Assuming df is your dataframe and 'array_column' is the column containing np.array of strings
    array_column_as_lists = df["models"].apply(
        lambda x: np.array(["None"]) if np.ndim(x) == 0 else x
    )
    # Now, flatten all arrays into one list
    flattened_strings = np.concatenate(array_column_as_lists.values)
    # Get unique strings
    unique_strings = np.unique(flattened_strings)
    # Convert to a list if needed
    unique_strings_list = unique_strings.tolist()
    models = gr.Dropdown(
        unique_strings_list,
        label="Search by Model",
        multiselect=True,
    )

    # Assuming df is your dataframe and 'array_column' is the column containing np.array of strings
    array_column_as_lists = df["datasets"].apply(
        lambda x: np.array(["None"]) if np.ndim(x) == 0 else x
    )

    # Now, flatten all arrays into one list
    flattened_strings = np.concatenate(array_column_as_lists.values)
    # Get unique strings
    unique_strings = np.unique(flattened_strings)
    # Convert to a list if needed
    unique_strings_list = unique_strings.tolist()
    datasets = gr.Dropdown(
        unique_strings_list,
        label="Search by Model",
        multiselect=True,
    )

    devMode = gr.Checkbox(value=False, label="DevMode Enabled")
    clear = gr.ClearButton(components=[emoji])

    df = pd.DataFrame(
        df[
            [
                "id",
                "emoji",
                "author",
                "url",
                "likes",
                "hardware",
                "sdk_tags",
                "models",
                "datasets",
            ]
        ]
    )
    df["url"] = df["url"].apply(
        lambda x: (
            f"<a target='_blank' href=https://huggingface.co/spaces/{x}>{x}</a>"
            if x is not None and "/" in x
            else f"<a target='_blank' href=https://{x[0]}>{x[0]}</a>"
        )
    )
    gr.DataFrame(
        filtered_df,
        inputs=[emoji, likes, author, hardware, sdk_tags, models, datasets],
        datatype="html",
    )


demo.launch()
