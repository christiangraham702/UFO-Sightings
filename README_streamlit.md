# UFO Sightings Analysis Dashboard

An interactive Streamlit dashboard for exploring and visualizing UFO sighting data patterns.

![UFO Dashboard](https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png)

## Overview

This Streamlit app provides an interactive visualization of UFO sighting data, allowing users to explore:

- Geographic distribution of UFO sightings with interactive maps
- Temporal patterns (yearly trends, monthly/hourly distributions)
- Common UFO shapes and how they've changed over time
- Text analysis of sighting descriptions
- Key statistics and insights about the data

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the App

To start the Streamlit application, run:

```bash
streamlit run app.py
```

This will start a local server and open the app in your default web browser, typically at http://localhost:8501.

## Features

The dashboard is organized into five main sections:

1. **Geographic Analysis**: Interactive maps showing the global distribution of UFO sightings
2. **Temporal Patterns**: Charts displaying when UFO sightings occur (by year, month, day, and hour)
3. **UFO Shapes**: Visualization of the most commonly reported UFO shapes
4. **Text Analysis**: Word cloud and sample descriptions from UFO sightings
5. **Summary Statistics**: Key metrics and insights about the dataset

## Data

The application uses the UFO sightings dataset located in `notebook/data/` (either the CSV or Excel version).

## Requirements

Main dependencies include:
- streamlit
- pandas
- matplotlib
- seaborn
- folium
- plotly
- wordcloud
- streamlit-folium

## Notes

This dashboard presents analysis of reported UFO sightings without making claims about their authenticity. It's designed for data exploration and visualization purposes only. 