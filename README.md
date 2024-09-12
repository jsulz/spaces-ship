---
title: Spaces Ship
emoji: 🚀
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 4.42.0
app_file: app.py
pinned: false
license: mit
short_description: See detailed information about all Spaces across the Hub.
---

# Spaces Ship

This is a spaceship through Spaces.

I started this mostly as a way to see more Spaces that I was interested in. Since there aren't any search/filtering options outside of full-text search and searching for Space titles, I wanted more ways to look around and get inspired.

It expanded as I saw what information you can get from leveraging the APIs in the `huggingface_hub` client.

Short-term, I'm running a lot of this locally, but long-term my goal is to run [this script](https://github.com/jsulz/hf-spaces-stats-builder/blob/main/src/pipeline.py) every 2 weeks, which:

- Calls `list_spaces` to get all spaces and some high level metadata
- Calls `space_info` to get the next level of depth from each space
- Stores this into a Dataset on the Hub - [jsulz/space-stats](https://huggingface.co/datasets/jsulz/space-stats)
  - Inspiration from this came from [cfahlgren1/hub-stats](cfahlgren1/hub-stats), but desiring one level of additional information (only available by making a lot of API calls)

I want this to be on a semi-regular cadence, but also respect that this takes in the realm of 12-15 hours (with some potential speedup from parallel )

This Space consumes that dataset into a Gradio app that has two tabs:

- Spaces Overview
- Spaces Search

The remaining content from here on out is a breakdown of what's in the Space, both tabs, and my feelings/thoughts about them after doing some digging.

# General

All of this needs context needs to live in the app in some form alongside the component. Avoiding that for the moment.

All of the labels and words that do exist need cleanup. Not worried about that for the moment.

# Spaces Overview

Charts exist for the following (commentary for each in sub-bullets):

- Growth of Spaces over Time
  - This is a line chart that shows the number of spaces created over time. Shows all Spaces, regardless of status.
- Distribution of Spaces by SDK
  - This is a pie chart that shows the distribution of Spaces by SDK. Can be either gradio, streamlit, docker, or static.
- Distribution of Spaces by Emoji
  - This is a pie chart that shows the distribution of Spaces by Emoji. This is a bit silly, but could be fun to work on this more to make it visually funny/appealing.
- Relationship between Number of Spaces Created and Number of Likes
  - This is a scatter plot that shows the relationship between the number of spaces created by an author and the number of likes. Not very interesting except for the outliers.
- Relationship between Space Emoji and Number of Likes
  - This is a scatter plot that shows the relationship between the emoji used in a space and the number of likes. Similar take as with the other scatter plot.
- Hardware in Use
  - This is a log scale bar chart of hardware in use. More interesting stuff here.
- Most Popular Model Authors
  - Bar chart of most popular model authors whose models are used in Spaces.
- Most Used Models
  - Bar chart of most popular models used in Spaces.
- Most Popular Dataset Authors
  - Bar chart of most popular dataset authors whose models are used in Spaces.
- Most Used Datasets
  - Bar chart of most popular datasets used in Spaces.
- Number of Duplicates by Space
  - Table showing the most duplicated Spaces.
- Number of Likes by Space
  - Table showing the most liked Spaces.
- Number of Spaces by Author
  - Table showing the most prolific Spaces authors.
- Number of Likes by Author
  - Table showing the authors with the most cumulative likes across all Spaces.

# Spaces Search

Filtration Options exist for the following (commentary for each in sub-bullets)

- Emojis
  - Fun, not very useful.
- Likes
  - Easy and helpful to see popular stuff.
- Authors
  - Kinda fun, but so many authors with so little context.
- SDK/Tags
  - Too many tags - lots of one-offs. Would maybe limit this to the top 10ish.
- Hardware
  - More useful than I thought it would be.
- License
  - Meh.
- Models
  - Very cool, but lots of one-offs and not highly used. Would maybe limit this to the top 10ish.
- Datasets
  - Same as models.
- Dev Mode
  - The interesting thing about this is how little it's used.
