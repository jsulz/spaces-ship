import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
from datasets import load_dataset


def load_transform_data():
    """
    Load and transform data from a parquet file.

    Returns:
        pandas.DataFrame: Transformed dataframe.
    """
    spaces_dataset = "jsulz/space-stats"
    dataset = load_dataset(spaces_dataset)
    df = dataset["train"].to_pandas()
    # combine the sdk and tags columns, one of which is a string and the other is an array of strings
    df["sdk"] = df["sdk"].apply(lambda x: np.array([str(x)]))
    df["licenses"] = df["license"].apply(
        lambda x: np.array([str(x)]) if x is None else x
    )
    # then combine the sdk and tags columns so that their elements are together
    df["sdk_tags"] = df[["sdk", "tags"]].apply(
        lambda x: np.concatenate((x.iloc[0], x.iloc[1])), axis=1
    )

    # Fill the NaN values with an empty string
    df["emoji"] = np.where(df["emoji"].isnull(), "", df["emoji"])

    # where the custom_domains column is not null, use that as the url, otherwise, use the host column
    df["url"] = np.where(
        df["custom_domains"].isnull(),
        df["id"],
        df["custom_domains"],
    )

    # Build up a pretty url that's clickable with the emoji
    df["url"] = df[["url", "emoji"]].apply(
        lambda x: (
            f'<a target="_blank" href=https://huggingface.co/spaces/{x.iloc[0]}>{str(x.iloc[1]) + " " + x.iloc[0]}</a>'
            if x.iloc[0] is not None
            else f'<a target="_blank" href=https://{x.iloc[0][0]}>{str(x.iloc[1]) + " " + x.iloc[0][0]}</a>'
        ),
        axis=1,
    )

    # Prep the models, datasets, and licenses columns for display
    df["r_models"] = [
        ", ".join(models) if models is not None else "" for models in df["models"]
    ]
    df["r_sdk_tags"] = [
        ", ".join(sdk_tags) if sdk_tags is not None else ""
        for sdk_tags in df["sdk_tags"]
    ]
    df["r_datasets"] = [
        ", ".join(datasets) if datasets is not None else ""
        for datasets in df["datasets"]
    ]
    df["r_licenses"] = [
        ", ".join(licenses) if licenses is not None else ""
        for licenses in df["licenses"]
    ]
    return df


def filtered_df(
    filtered_emojis,
    filtered_likes,
    filtered_author,
    filtered_hardware,
    filtered_tags,
    filtered_models,
    filtered_datasets,
    space_licenses,
    filtered_devmode,
):
    """
    Filter the dataframe based on the given criteria.

    Args:
        filtered_emojis (list): List of emojis to filter the dataframe by.
        filtered_likes (int): Minimum number of likes to filter the dataframe by.
        filtered_author (list): List of authors to filter the dataframe by.
        filtered_hardware (list): List of hardware to filter the dataframe by.
        filtered_tags (list): List of tags to filter the dataframe by.
        filtered_models (list): List of models to filter the dataframe by.
        filtered_datasets (list): List of datasets to filter the dataframe by.
        space_licenses (list): List of licenses to filter the dataframe by.

    Returns:
        pandas.DataFrame: Filtered dataframe with the following columns: "URL", "Likes", "Models", "Datasets", "Licenses".
    """
    _df = df
    if filtered_emojis:
        _df = _df[_df["emoji"].isin(filtered_emojis)]
    if filtered_likes:
        _df = _df[_df["likes"] >= filtered_likes]
    if filtered_author:
        _df = _df[_df["author"].isin(filtered_author)]
    if filtered_hardware:
        _df = _df[_df["hardware"].isin(filtered_hardware)]
    if filtered_tags:
        _df = _df[
            _df["sdk_tags"].apply(lambda x: any(tag in x for tag in filtered_tags))
        ]
    if filtered_models:
        _df = _df[
            _df["models"].apply(
                lambda x: (
                    any(model in x for model in filtered_models)
                    if x is not None
                    else False
                )
            )
        ]
    if filtered_datasets:
        _df = _df[
            _df["datasets"].apply(
                lambda x: (
                    any(dataset in x for dataset in filtered_datasets)
                    if x is not None
                    else False
                )
            )
        ]
    if space_licenses:
        _df = _df[
            _df["licenses"].apply(
                lambda x: (
                    any(space_license in x for space_license in space_licenses)
                    if x is not None
                    else False
                )
            )
        ]

    # rename the columns names to make them more readable
    _df = _df.rename(
        columns={
            "url": "URL",
            "likes": "Likes",
            "r_models": "Models",
            "r_datasets": "Datasets",
            "r_licenses": "Licenses",
        }
    )
    if filtered_devmode:
        _df = _df[_df["devMode"] == filtered_devmode]

    return _df[["URL", "Likes", "Models", "Datasets", "Licenses"]]


def count_items(items):
    """
    Count the occurrences of items and authors in a given list of items.
    Parameters:
    items (dataframe column): A dataframe column containing a list of items.
    Returns:
    tuple: A tuple containing two dictionaries. The first dictionary contains the count of each item,
    and the second dictionary contains the count of each author.
    """
    items = np.concatenate([arr for arr in items.values if arr is not None])
    item_count = {}
    item_author_count = {}
    for item in items:
        if item in item_count:
            item_count[item] += 1
        else:
            item_count[item] = 1
        author = item.split("/")[0]
        if author in item_author_count:
            item_author_count[author] += 1
        else:
            item_author_count[author] = 1

    return item_count, item_author_count


def flatten_column(_df, column):
    """
    Flattens a column in a DataFrame.

    Args:
        _df (pandas.DataFrame): The DataFrame containing the column.
        column (str): The name of the column to flatten.

    Returns:
        list: A list of unique values from the flattened column.
    """
    column_to_list = _df[column].apply(
        lambda x: np.array(["None"]) if np.ndim(x) == 0 else x
    )
    flattened = np.concatenate(column_to_list.values)
    uniques = np.unique(flattened)
    return uniques.tolist()


with gr.Blocks(fill_width=True) as demo:
    df = load_transform_data()
    with gr.Tab(label="Spaces Overview"):

        # The Pandas dataframe has a datetime column. Plot the growth of spaces (row entries) over time.
        # The x-axis should be the date and the y-axis should be the cumulative number of spaces created up to that date .
        df = df.sort_values("created_at")
        df["cumulative_spaces"] = df["created_at"].rank(method="first").astype(int)
        fig1 = px.line(
            df,
            x="created_at",
            y="cumulative_spaces",
            title="Growth of Spaces Over Time",
            labels={"created_at": "Date", "cumulative_spaces": "Number of Spaces"},
            template="plotly_dark",
        )
        gr.Plot(fig1)

        with gr.Row():
            # Create a pie charge showing the distribution of spaces by SDK
            fig2 = px.pie(
                df,
                names="sdk",
                title="Distribution of Spaces by SDK",
                template="plotly_dark",
            )
            gr.Plot(fig2)

            # create a pie chart showing the distribution of spaces by emoji for the top 10 used emojis
            emoji_counts = df["emoji"].value_counts().head(10).reset_index()
            fig3 = px.pie(
                emoji_counts,
                names="emoji",
                values="count",
                title="Distribution of Spaces by Emoji",
                template="plotly_dark",
            )
            gr.Plot(fig3)

        # Create a scatter plot showing the relationship between the number of likes and the number of spaces created by an author
        author_likes = (
            df.groupby("author").agg({"likes": "sum", "id": "count"}).reset_index()
        )
        fig4 = px.scatter(
            author_likes,
            x="id",
            y="likes",
            title="Relationship between Number of Spaces Created and Number of Likes",
            labels={"id": "Number of Spaces Created", "likes": "Number of Likes"},
            hover_data={"author": True},
            template="plotly_dark",
        )
        gr.Plot(fig4)

        # Create a scatter plot showing the relationship between the number of likes and the number of spaces created by an author
        emoji_likes = (
            df.groupby("emoji")
            .agg({"likes": "sum", "id": "count"})
            .sort_values(by="likes", ascending=False)
            .head(20)
            .reset_index()
        )
        fig10 = px.scatter(
            emoji_likes,
            x="id",
            y="likes",
            title="Relationship between Space Emoji and Number of Likes",
            labels={"id": "Number of Spaces Created", "likes": "Number of Likes"},
            hover_data={"emoji": True},
            template="plotly_dark",
        )
        gr.Plot(fig10)

        # Create a bar chart of hardware in use
        hardware = df["hardware"].value_counts().reset_index()
        hardware.columns = ["Hardware", "Number of Spaces"]
        fig5 = px.bar(
            hardware,
            x="Hardware",
            y="Number of Spaces",
            title="Hardware in Use",
            labels={
                "Hardware": "Hardware",
                "Number of Spaces": "Number of Spaces (log scale)",
            },
            color="Hardware",
            template="plotly_dark",
        )
        fig5.update_layout(yaxis_type="log")
        gr.Plot(fig5)

        model_count, model_author_count = count_items(df["models"])
        model_author_count = pd.DataFrame(
            model_author_count.items(), columns=["Model Author", "Number of Spaces"]
        )
        fig8 = px.bar(
            model_author_count.sort_values("Number of Spaces", ascending=False).head(
                20
            ),
            x="Model Author",
            y="Number of Spaces",
            title="Most Popular Model Authors",
            labels={"Model": "Model", "Number of Spaces": "Number of Spaces"},
            template="plotly_dark",
        )
        gr.Plot(fig8)
        model_count = pd.DataFrame(
            model_count.items(), columns=["Model", "Number of Spaces"]
        )
        # then make a bar chart
        fig6 = px.bar(
            model_count.sort_values("Number of Spaces", ascending=False).head(20),
            x="Model",
            y="Number of Spaces",
            title="Most Used Models",
            labels={"Model": "Model", "Number of Spaces": "Number of Spaces"},
            template="plotly_dark",
        )
        gr.Plot(fig6)

        dataset_count, dataset_author_count = count_items(df["datasets"])
        dataset_count = pd.DataFrame(
            dataset_count.items(), columns=["Datasets", "Number of Spaces"]
        )
        dataset_author_count = pd.DataFrame(
            dataset_author_count.items(), columns=["Dataset Author", "Number of Spaces"]
        )
        fig9 = px.bar(
            dataset_author_count.sort_values("Number of Spaces", ascending=False).head(
                20
            ),
            x="Dataset Author",
            y="Number of Spaces",
            title="Most Popular Dataset Authors",
            labels={
                "Dataset Author": "Dataset Author",
                "Number of Spaces": "Number of Spaces",
            },
            template="plotly_dark",
        )
        gr.Plot(fig9)
        # then make a bar chart
        fig7 = px.bar(
            dataset_count.sort_values("Number of Spaces", ascending=False).head(20),
            x="Datasets",
            y="Number of Spaces",
            title="Most Used Datasets",
            labels={"Datasets": "Datasets", "Number of Spaces": "Number of Spaces"},
            template="plotly_dark",
        )
        gr.Plot(fig7)

        with gr.Row():
            # Get the most duplicated spaces
            duplicated_spaces = (
                df["duplicated_from"].value_counts().head(20).reset_index()
            )
            duplicated_spaces["duplicated_from"] = duplicated_spaces[
                "duplicated_from"
            ].apply(
                lambda x: f"<a target='_blank' href=https://huggingface.co/spaces/{x}>{x}</a>"
            )
            duplicated_spaces.columns = ["Space", "Number of Duplicates"]
            gr.DataFrame(duplicated_spaces, datatype="html")

            # Get the most liked spaces
            liked_spaces = (
                df[["id", "likes"]].sort_values(by="likes", ascending=False).head(20)
            )
            liked_spaces["id"] = liked_spaces["id"].apply(
                lambda x: f"<a target='_blank' href=https://huggingface.co/spaces/{x}>{x}</a>"
            )
            liked_spaces.columns = ["Space", "Number of Likes"]
            gr.DataFrame(liked_spaces, datatype="html")

        with gr.Row():
            # Create a dataframe with the top 10 authors and the number of spaces they have created
            author_counts = df["author"].value_counts().head(20).reset_index()
            author_counts["author"] = author_counts["author"].apply(
                lambda x: f"<a target='_blank' href=https://huggingface.co/{x}>{x}</a>"
            )
            author_counts.columns = ["Author", "Number of Spaces"]
            gr.DataFrame(author_counts, datatype="html")

            # create a dataframe where we groupby author and sum their likes
            author_likes = df.groupby("author").agg({"likes": "sum"}).reset_index()
            author_likes = author_likes.sort_values(by="likes", ascending=False).head(
                20
            )
            author_likes["author"] = author_likes["author"].apply(
                lambda x: f"<a target='_blank' href=https://huggingface.co/{x}>{x}</a>"
            )
            author_likes.columns = ["Author", "Number of Likes"]
            gr.DataFrame(author_likes, datatype="html")

    with gr.Tab(label="Spaces Search"):
        df = df[df["stage"] == "RUNNING"]

        # Layout
        with gr.Row():
            emoji = gr.Dropdown(
                df["emoji"].unique().tolist(),
                label="Search by Emoji 🤗",
                multiselect=True,
            )  # Dropdown to select the emoji
            likes = gr.Slider(
                minimum=df["likes"].min(),
                maximum=df["likes"].max(),
                step=1,
                label="Filter by Likes",
            )  # Slider to filter by likes
        with gr.Row():
            author = gr.Dropdown(
                df["author"].unique().tolist(),
                label="Search by Author",
                multiselect=True,
            )
            # get the list of unique strings in the sdk_tags column
            sdk_tags = np.unique(np.concatenate(df["sdk_tags"].values))
            # create a dropdown for the sdk_tags
            sdk_tags = gr.Dropdown(
                sdk_tags.tolist(), label="Filter by SDK/Tags", multiselect=True
            )
        with gr.Row():
            # create a gradio checkbox group for hardware
            hardware = gr.CheckboxGroup(
                df["hardware"].unique().tolist(), label="Filter by Hardware"
            )

            licenses = np.unique(np.concatenate(df["licenses"].values))
            space_license = gr.Dropdown(licenses.tolist(), label="Filter by license")

        with gr.Row():
            models = gr.Dropdown(
                flatten_column(df, "models"),
                label="Search by Model",
                multiselect=True,
            )
            datasets = gr.Dropdown(
                flatten_column(df, "datasets"),
                label="Search by Dataset",
                multiselect=True,
            )

        devmode = gr.Checkbox(label="Show Dev Mode Spaces")
        clear = gr.ClearButton(
            components=[
                emoji,
                author,
                hardware,
                sdk_tags,
                models,
                datasets,
                space_license,
            ]
        )

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
                    "licenses",
                    "r_sdk_tags",
                    "r_models",
                    "r_datasets",
                    "r_licenses",
                    "devMode",
                ]
            ]
        )
        gr.DataFrame(
            filtered_df,
            inputs=[
                emoji,
                likes,
                author,
                hardware,
                sdk_tags,
                models,
                datasets,
                space_license,
                devmode,
            ],
            datatype="html",
            wrap=True,
            column_widths=["25%", "5%", "25%", "25%", "20%"],
        )


demo.launch(share=True)
