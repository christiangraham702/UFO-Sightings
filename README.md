# UFO Sightings Data Analysis

This project analyzes a dataset of UFO sightings, exploring patterns in time, geography, and reported characteristics.

## Dataset

The dataset contains UFO sighting reports with the following information:
- Date and time of sighting
- Location (country, city, state, latitude, longitude)
- Shape of the UFO
- Summary description of the sighting

## Key Findings

After analyzing the UFO sightings dataset, we discovered several interesting patterns:

### Temporal Patterns
- All data is from 2016, with 5,177 sightings in total
- Most sightings occur in July (539 sightings)
- Nighttime sightings are much more common (81.5% occur between 6 PM and 6 AM)
- Peak hour for sightings is 9 PM (21:00) with 845 sightings
- Sunday has the most sightings (779)

### Geographic Distribution
- USA has the vast majority of reports (5,027 out of 5,177 sightings, or 97%)
- Canada is a distant second with 150 sightings
- Within the US, California has the most sightings (546), followed by other states

### UFO Characteristics
- The most commonly reported UFO shape is "Light" (1,118 sightings)
- Other common shapes include Circle (678), Triangle (498), Unknown (431), and Fireball (430)
- The interactive visualizations reveal geographic clusters of sightings

## Features

The analysis includes:

1. **Temporal Analysis**
   - Yearly trends in UFO sightings
   - Monthly patterns
   - Time of day distribution
   - Day of week analysis

2. **Geographic Analysis**
   - Distribution by country
   - US state distribution
   - Interactive heatmap of sighting locations

3. **Shape Analysis**
   - Most commonly reported shapes
   - Shape trends over time

4. **Text Analysis**
   - Word cloud from sighting descriptions
   - Common themes in reports

5. **Interactive Visualizations**
   - Interactive map with detailed popups
   - Density map showing hotspots

## Getting Started

### Prerequisites

Install the required packages:

```bash
# Create a virtual environment (recommended)
python3 -m venv ufo_env
source ufo_env/bin/activate  # On Windows: ufo_env\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Running the Analysis Script

To run the full analysis script:

```bash
python ufo_analysis.py
```

The script will generate visualizations and save them to a `visualizations` directory.

### Interactive Streamlit Dashboard

The project now includes an interactive Streamlit dashboard for exploring the UFO sightings data in real-time.

To launch the Streamlit app:

```bash
streamlit run app.py
```

Or use the provided convenience scripts:
- On Unix/Mac: `./run_app.sh`
- On Windows: Double-click `run_app.bat`

The Streamlit dashboard provides:
- Interactive maps showing the geographic distribution of sightings
- Dynamic visualizations of temporal patterns
- UFO shape analysis with trends over time
- Text analysis of sighting descriptions
- Summary statistics and insights

#### Features of the Streamlit App

The app organizes the analysis into five main tabs:
1. **Geographic Analysis**: Interactive maps with heatmaps and markers
2. **Temporal Patterns**: Visualizations of when UFO sightings occur
3. **UFO Shapes**: Analysis of commonly reported shapes
4. **Text Analysis**: Word cloud and sample sighting descriptions
5. **Summary Statistics**: Key metrics and overview of the dataset

### Viewing the Static Dashboard

The project also includes a static dashboard that displays all the visualizations in a user-friendly web interface.

To view the static dashboard:

1. Make sure you've run the analysis script first (`python ufo_analysis.py`)
2. Open the index.html file in a web browser:
   - **On macOS**: Run `open index.html` in terminal
   - **On Windows**: Double-click the index.html file or run `start index.html` in command prompt
   - **On Linux**: Run `xdg-open index.html` in terminal

## Visualization Examples

All visualizations are saved to the `visualizations` directory:

- Static images (PNG format):
  - Yearly trends
  - Monthly patterns
  - Shape distributions
  - Word cloud of descriptions

- Interactive visualizations (HTML format):
  - UFO sightings map with markers and popups
  - Density heatmap

## Insights

The analysis reveals that UFO sightings follow certain patterns - they're mostly reported at night, particularly during summer months, and tend to be concentrated in specific geographic regions like California. The most common description is simply "Light" which suggests many sightings may be related to atmospheric phenomena, aircraft, or celestial objects viewed under certain conditions.

## Deployment

The Streamlit app can be deployed to platforms like:
- [Streamlit Cloud](https://streamlit.io/cloud)
- Heroku
- Any server that supports Python web applications

## Notes

This is an exploratory data analysis project and does not attempt to validate the authenticity of the reported sightings. 