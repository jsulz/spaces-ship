import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
# Load the spaces.parquet file as a dataframe and do some pre cleaning steps


"""
Todos:
    Clean up existing filtering code
"""


def filtered_df(emoji, likes, author, hardware, tags, models, datasets, space_licenses):
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
            'url': 'URL',
            'likes': 'Likes',
            "r_models": "Models",
            "r_datasets": "Datasets",
            "r_licenses": "Licenses",
        }
    )

    return _df[["URL", "Likes", "Models", "Datasets", "Licenses" ]]


with gr.Blocks(fill_width=True) as demo:
    with gr.Tab(label="Spaces Overview"):

        # The Pandas dataframe has a datetime column. Plot the growth of spaces (row entries) over time. 
        # The x-axis should be the date and the y-axis should be the cumulative number of spaces created up to that date .
        df = pd.read_parquet("spaces.parquet")
        df = df.sort_values("created_at")
        df['cumulative_spaces'] = df['created_at'].rank(method='first').astype(int)
        fig1 = px.line(df, x='created_at', y='cumulative_spaces', title='Growth of Spaces Over Time', labels={'created_at': 'Date', 'cumulative_spaces': 'Number of Spaces'}, template='plotly_dark')
        gr.Plot(fig1)

        # Create a pie charge showing the distribution of spaces by SDK
        fig2 = px.pie(df, names='sdk', title='Distribution of Spaces by SDK', template='plotly_dark')
        gr.Plot(fig2)

        # create a pie chart showing the distribution of spaces by emoji for the top 10 used emojis
        emoji_counts = df['emoji'].value_counts().head(10).reset_index()
        fig3 = px.pie(emoji_counts, names='emoji', values='count', title='Distribution of Spaces by Emoji', template='plotly_dark')
        gr.Plot(fig3)

        # Create a dataframe with the top 10 authors and the number of spaces they have created
        author_counts = df['author'].value_counts().head(20).reset_index()
        author_counts.columns = ['Author', 'Number of Spaces']
        gr.DataFrame(author_counts)

        # Create a scatter plot showing the relationship between the number of likes and the number of spaces created by an author
        author_likes = df.groupby('author').agg({'likes': 'sum', 'id': 'count'}).reset_index()
        fig4 = px.scatter(author_likes, x='id', y='likes', title='Relationship between Number of Spaces Created and Number of Likes', labels={'id': 'Number of Spaces Created', 'likes': 'Number of Likes'}, hover_data={'author': True}, template='plotly_dark')
        gr.Plot(fig4)

        # Create a scatter plot showing the relationship between the number of likes and the number of spaces created by an author
        emoji_likes = df.groupby('emoji').agg({'likes': 'sum', 'id': 'count'}).sort_values(by='likes', ascending=False).head(20).reset_index()
        fig10 = px.scatter(emoji_likes, x='id', y='likes', title='Relationship between Number of Spaces Created and Number of Likes', labels={'id': 'Number of Spaces Created', 'likes': 'Number of Likes'}, hover_data={'emoji': True}, template='plotly_dark')
        gr.Plot(fig10)

        # Create a bar chart of hardware in use
        hardware = df['hardware'].value_counts().reset_index()
        hardware.columns = ['Hardware', 'Number of Spaces']
        fig5 = px.bar(hardware, x='Hardware', y='Number of Spaces', title='Hardware in Use', labels={'Hardware': 'Hardware', 'Number of Spaces': 'Number of Spaces (log scale)'}, color='Hardware', template='plotly_dark')
        fig5.update_layout(yaxis_type='log')
        gr.Plot(fig5)

        models = np.concatenate([arr for arr in df['models'].values if arr is not None])
        model_count = {}
        model_author_count = {}
        for model in models:
            author = model.split('/')[0]
            if model in model_count:
                model_count[model] += 1
            else:
                model_count[model] = 1
            if author in model_author_count:
                model_author_count[author] += 1
            else:
                model_author_count[author] = 1
        model_author_count = pd.DataFrame(model_author_count.items(), columns=['Model Author', 'Number of Spaces'])
        fig8 = px.bar(model_author_count.sort_values('Number of Spaces', ascending=False).head(20), x='Model Author', y='Number of Spaces', title='Most Popular Model Authors', labels={'Model': 'Model', 'Number of Spaces': 'Number of Spaces'}, template='plotly_dark')
        gr.Plot(fig8)
        model_count = pd.DataFrame(model_count.items(), columns=['Model', 'Number of Spaces'])
        # then make a bar chart
        fig6 = px.bar(model_count.sort_values('Number of Spaces', ascending=False).head(20), x='Model', y='Number of Spaces', title='Most Used Models', labels={'Model': 'Model', 'Number of Spaces': 'Number of Spaces'}, template='plotly_dark')
        gr.Plot(fig6)

        datasets = np.concatenate([arr for arr in df['datasets'].values if arr is not None])
        dataset_count = {}
        dataset_author_count = {}
        for dataset in datasets:
            author = dataset.split('/')[0]
            if dataset in dataset_count:
                dataset_count[dataset] += 1
            else:
                dataset_count[dataset] = 1
            if author in dataset_author_count:
                dataset_author_count[author] += 1
            else:
                dataset_author_count[author] = 1
        dataset_count = pd.DataFrame(dataset_count.items(), columns=['Datasets', 'Number of Spaces'])
        dataset_author_count = pd.DataFrame(dataset_author_count.items(), columns=['Dataset Author', 'Number of Spaces'])
        fig9 = px.bar(dataset_author_count.sort_values('Number of Spaces', ascending=False).head(20), x='Dataset Author', y='Number of Spaces', title='Most Popular Dataset Authors', labels={'Dataset Author': 'Dataset Author', 'Number of Spaces': 'Number of Spaces'}, template='plotly_dark')
        gr.Plot(fig9)
        # then make a bar chart
        fig7 = px.bar(dataset_count.sort_values('Number of Spaces', ascending=False).head(20), x='Datasets', y='Number of Spaces', title='Most Used Datasets', labels={'Datasets': 'Datasets', 'Number of Spaces': 'Number of Spaces'}, template='plotly_dark')
        gr.Plot(fig7)

        # Get the most duplicated spaces
        duplicated_spaces = df['duplicated_from'].value_counts().head(20).reset_index()
        duplicated_spaces.columns = ['Space', 'Number of Duplicates']
        gr.DataFrame(duplicated_spaces)

        # Get the most duplicated spaces
        liked_spaces = df[['id', 'likes']].sort_values(by='likes', ascending=False).head(20)
        liked_spaces.columns = ['Space', 'Number of Likes']
        gr.DataFrame(liked_spaces)

        # Get the spaces with the longest READMEs
        readme_sizes = df[['id', 'readme_size']].sort_values(by='readme_size', ascending=False).head(20)
        readme_sizes.columns = ['Space', 'Longest READMEs']
        gr.DataFrame(readme_sizes)
        
    with gr.Tab(label="Spaces Search"):
        df = pd.read_parquet("spaces.parquet")
        df = df[df["stage"] == "RUNNING"]
        # combine the sdk and tags columns, one of which is a string and the other is an array of strings
        # first convert the sdk column to an array of strings
        df["sdk"] = df["sdk"].apply(lambda x: np.array([str(x)]))
        df["licenses"] = df["license"].apply(
            lambda x: np.array([str(x)]) if x is None else x
        )
        # then combine the sdk and tags columns so that their elements are together
        df["sdk_tags"] = df[["sdk", "tags"]].apply(
            lambda x: np.concatenate((x.iloc[0], x.iloc[1])), axis=1
        )

        df['emoji'] = np.where(df['emoji'].isnull(), '', df['emoji'])

        # where the custom_domains column is not null, use that as the url, otherwise, use the host column
        df["url"] = np.where(
            df["custom_domains"].isnull(),
            df["id"],
            df["custom_domains"],
        )
        df["url"] = df[["url", "emoji"]].apply(
            lambda x: (
                f"<a target='_blank' href=https://huggingface.co/spaces/{x.iloc[0]}>{str(x.iloc[1]) + " " + x.iloc[0]}</a>"
                if x.iloc[0] is not None and "/" in x.iloc[0]
                else f"<a target='_blank' href=https://{x.iloc[0][0]}>{str(x.iloc[1]) + " " + x.iloc[0][0]}</a>"
            ),
            axis=1,
        )

        # Make all of this human readable
        df["r_models"] = [', '.join(models) if models is not None else '' for models in df["models"]]
        df["r_sdk_tags"] = [', '.join(sdk_tags) if sdk_tags is not None else '' for sdk_tags in df["sdk_tags"]]
        df["r_datasets"] = [', '.join(datasets) if datasets is not None else '' for datasets in df["datasets"]]
        df["r_licenses"] = [', '.join(licenses) if licenses is not None else '' for licenses in df["licenses"]]


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

        licenses = np.unique(np.concatenate(df["licenses"].values))
        space_license = gr.CheckboxGroup(licenses.tolist(), label="Filter by license")

        # If the models column is none make it an array of "none" so that things don't break
        models_column_to_list = df["models"].apply(
            lambda x: np.array(["None"]) if np.ndim(x) == 0 else x
        )
        # Now, flatten all arrays into one list
        models_flattened = np.concatenate(models_column_to_list.values)
        # Get unique strings
        unique_models = np.unique(models_flattened)
        models = gr.Dropdown(
            unique_models.tolist(),
            label="Search by Model",
            multiselect=True,
        )

        # Do the same for datasets that we did for models
        datasets_column_to_list = df["datasets"].apply(
            lambda x: np.array(["None"]) if np.ndim(x) == 0 else x
        )
        flattened_datasets = np.concatenate(datasets_column_to_list.values)
        unique_datasets = np.unique(flattened_datasets)
        datasets = gr.Dropdown(
            unique_datasets.tolist(),
            label="Search by Dataset",
            multiselect=True,
        )

        devMode = gr.Checkbox(value=False, label="DevMode Enabled")
        clear = gr.ClearButton(components=[
                emoji,
                author,
                hardware,
                sdk_tags,
                models,
                datasets,
                space_license
                ])

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
            ],
            datatype="html",
            wrap=True, 
            column_widths=["25%", "5%", "25%", "25%", "20%"]
        )


demo.launch()
