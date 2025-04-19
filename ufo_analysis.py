import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from collections import Counter
from wordcloud import WordCloud
import folium
from folium.plugins import HeatMap, MarkerCluster
import re
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Load data
print("Loading UFO sightings dataset...")
try:
    df = pd.read_csv('notebook/data/UFOs_coord.csv')
    print(f"Successfully loaded dataset with {df.shape[0]} records.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    print("Trying Excel file instead...")
    try:
        df = pd.read_excel('notebook/data/UFOs_coord.xlsx')
        print(f"Successfully loaded Excel dataset with {df.shape[0]} records.")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        exit(1)

# Display basic information
print("\n--- Dataset Overview ---")
print(df.info())
print("\n--- First 5 records ---")
print(df.head())

# Data preprocessing
print("\n--- Data Preprocessing ---")

# Check for missing values
print("\nMissing values by column:")
print(df.isnull().sum())

# Extract year from date
if 'Date / Time' in df.columns:
    print("\nExtracting time-related features...")
    # Try different date formats
    try:
        df['datetime'] = pd.to_datetime(df['Date / Time'])
    except:
        try:
            df['datetime'] = pd.to_datetime(df['Date / Time'], format='%m/%d/%y %H:%M')
        except:
            # Check if 'Year' column already exists
            if 'Year' not in df.columns:
                # Extract year using regex as fallback
                df['Year'] = df['Date / Time'].str.extract(r'(\d{4})$', expand=False)
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # If datetime conversion was successful
    if 'datetime' in df.columns:
        df['Year'] = df['datetime'].dt.year
        df['Month'] = df['datetime'].dt.month
        df['Hour'] = df['datetime'].dt.hour
        df['Day'] = df['datetime'].dt.day
        df['DayOfWeek'] = df['datetime'].dt.dayofweek
        
        # Create day/night indicator (approximation)
        df['is_night'] = ((df['Hour'] >= 18) | (df['Hour'] <= 5)).astype(int)
        
        print("Created temporal features: Year, Month, Hour, Day, DayOfWeek, is_night")

# Create a directory for saving visualizations
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')
    print("Created 'visualizations' directory")

# 1. Time-based Analysis
print("\n--- Time-based Analysis ---")

# Yearly trend
plt.figure(figsize=(12, 6))
if 'Year' in df.columns:
    yearly_counts = df['Year'].value_counts().sort_index()
    yearly_counts.plot(kind='line', marker='o', linewidth=2)
    plt.title('UFO Sightings by Year', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Sightings', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/yearly_trend.png')
    print(f"Peak year for UFO sightings: {yearly_counts.idxmax()} with {yearly_counts.max()} sightings")

# Monthly pattern
if 'Month' in df.columns:
    plt.figure(figsize=(12, 6))
    month_order = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_counts = df['Month'].value_counts().reindex(month_order)
    
    ax = sns.barplot(x=month_names, y=monthly_counts.values)
    plt.title('UFO Sightings by Month', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Number of Sightings', fontsize=14)
    
    # Add value labels to the bars
    for i, count in enumerate(monthly_counts.values):
        ax.text(i, count + 5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/monthly_pattern.png')
    print(f"Month with most UFO sightings: {month_names[monthly_counts.idxmax()-1]} with {monthly_counts.max()} sightings")

# Time of day
if 'Hour' in df.columns:
    plt.figure(figsize=(14, 6))
    hourly_counts = df['Hour'].value_counts().sort_index()
    
    # Create custom colors for day/night visualization
    colors = ['#3a6186' if i >= 18 or i <= 5 else '#89253e' for i in hourly_counts.index]
    
    ax = sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette=colors)
    plt.title('UFO Sightings by Hour of Day', fontsize=16)
    plt.xlabel('Hour (24-hour format)', fontsize=14)
    plt.ylabel('Number of Sightings', fontsize=14)
    plt.xticks(range(0, 24))
    
    # Add a legend
    import matplotlib.patches as mpatches
    night_patch = mpatches.Patch(color='#3a6186', label='Night (6 PM - 6 AM)')
    day_patch = mpatches.Patch(color='#89253e', label='Day (6 AM - 6 PM)')
    plt.legend(handles=[day_patch, night_patch])
    
    plt.tight_layout()
    plt.savefig('visualizations/hourly_pattern.png')
    print(f"Hour with most UFO sightings: {hourly_counts.idxmax()} with {hourly_counts.max()} sightings")
    
    # Check if sightings are more common at night
    if 'is_night' in df.columns:
        night_percentage = df['is_night'].mean() * 100
        print(f"Percentage of sightings occurring at night: {night_percentage:.2f}%")

# Day of week
if 'DayOfWeek' in df.columns:
    plt.figure(figsize=(12, 6))
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['DayOfWeek'].value_counts().reindex(range(7))
    
    ax = sns.barplot(x=day_names, y=day_counts.values)
    plt.title('UFO Sightings by Day of Week', fontsize=16)
    plt.xlabel('Day', fontsize=14)
    plt.ylabel('Number of Sightings', fontsize=14)
    plt.xticks(rotation=45)
    
    for i, count in enumerate(day_counts.values):
        ax.text(i, count + 5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/day_of_week_pattern.png')
    print(f"Day with most UFO sightings: {day_names[day_counts.idxmax()]} with {day_counts.max()} sightings")

# 2. Geographic Analysis
print("\n--- Geographic Analysis ---")

# Distribution by country
plt.figure(figsize=(12, 6))
country_counts = df['Country'].value_counts()
country_counts.plot(kind='bar')
plt.title('UFO Sightings by Country', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Number of Sightings', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/country_distribution.png')
print(f"Most UFO sightings reported in: {country_counts.index[0]} with {country_counts.values[0]} sightings")

# US State distribution (if applicable)
if 'State' in df.columns:
    usa_data = df[df['Country'] == 'USA']
    if not usa_data.empty:
        plt.figure(figsize=(14, 8))
        top_states = usa_data['State'].value_counts().head(15)
        top_states.plot(kind='bar')
        plt.title('Top 15 US States by UFO Sightings', fontsize=16)
        plt.xlabel('State', fontsize=14)
        plt.ylabel('Number of Sightings', fontsize=14)
        plt.tight_layout()
        plt.savefig('visualizations/us_state_distribution.png')
        print(f"US State with most UFO sightings: {top_states.index[0]} with {top_states.values[0]} sightings")

# Create an interactive map
print("\nCreating interactive map of UFO sightings...")
sightings_map = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], 
                           zoom_start=4, 
                           tiles='CartoDB dark_matter')

# Add a heatmap layer
heat_data = [[row['lat'], row['lng']] for _, row in df.iterrows()]
HeatMap(heat_data, radius=8, blur=10).add_to(sightings_map)

# Add a marker cluster for individual sightings
marker_cluster = MarkerCluster().add_to(sightings_map)

# Sample a subset for markers to avoid overloading the map
sample_size = min(500, len(df))
sampled_df = df.sample(sample_size, random_state=42)

for _, row in sampled_df.iterrows():
    # Create a popup with sighting details
    popup_text = f"""
    <b>Date:</b> {row['Date / Time']}<br>
    <b>Location:</b> {row['City']}, {row['State']}<br>
    <b>Shape:</b> {row['Shape']}<br>
    <b>Summary:</b> {row['Summary'][:100]}...
    """
    
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color='green', icon='alien', prefix='fa')
    ).add_to(marker_cluster)

# Save the map
sightings_map.save('visualizations/ufo_sightings_map.html')
print("Interactive map saved as 'visualizations/ufo_sightings_map.html'")

# 3. Shape Analysis
print("\n--- UFO Shape Analysis ---")

if 'Shape' in df.columns:
    plt.figure(figsize=(12, 8))
    shape_counts = df['Shape'].value_counts().head(10)
    ax = sns.barplot(x=shape_counts.values, y=shape_counts.index)
    plt.title('Top 10 Reported UFO Shapes', fontsize=16)
    plt.xlabel('Number of Sightings', fontsize=14)
    plt.ylabel('Shape', fontsize=14)
    
    # Add value labels
    for i, count in enumerate(shape_counts.values):
        ax.text(count + 10, i, str(count), va='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/ufo_shapes.png')
    print(f"Most common UFO shape: {shape_counts.index[0]} with {shape_counts.values[0]} sightings")
    
    # Check if shape prevalence has changed over time
    if 'Year' in df.columns:
        # Get top 5 shapes for analysis
        top_shapes = df['Shape'].value_counts().head(5).index.tolist()
        
        # Filter data for top shapes and create pivot table
        shape_by_year = df[df['Shape'].isin(top_shapes)].pivot_table(
            index='Year', 
            columns='Shape', 
            aggfunc='size', 
            fill_value=0
        )
        
        # Plot trends
        plt.figure(figsize=(14, 8))
        for shape in top_shapes:
            if shape in shape_by_year.columns:
                plt.plot(shape_by_year.index, shape_by_year[shape], marker='o', linewidth=2, label=shape)
        
        plt.title('UFO Shape Trends Over Time (Top 5 Shapes)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Sightings', fontsize=14)
        plt.legend(title='Shape')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/shape_trends.png')
        print("Created visualization of shape trends over time")

# 4. Text Analysis of Summaries
print("\n--- Summary Text Analysis ---")

if 'Summary' in df.columns:
    # Combine all summaries
    all_summaries = ' '.join(df['Summary'].dropna())
    
    # Clean text
    all_summaries = re.sub(r'[^\w\s]', '', all_summaries.lower())
    
    # Create a list of common words to exclude
    stopwords = ['the', 'and', 'of', 'to', 'a', 'in', 'that', 'was', 'i', 'it', 'for', 'on', 'with', 'as', 'at', 'by', 'from',
                 'be', 'this', 'have', 'or', 'had', 'but', 'not', 'what', 'all', 'were', 'when', 'we', 'there', 'can', 'an',
                 'my', 'they', 'no', 'is', 'about', 'our', 'has', 'very', 'would', 'me', 'which', 'him', 'them', 'then', 'she',
                 'who', 'her', 'out', 'do', 'their', 'so', 'up', 'could', 'get', 'been', 'just', 'your', 'how', 'more', 'if',
                 'some', 'its', 'any', 'will', 'did']
    
    # Create a word cloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='black',
        stopwords=stopwords,
        colormap='viridis',
        max_words=100
    ).generate(all_summaries)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in UFO Sighting Descriptions', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/summary_wordcloud.png')
    print("Created word cloud from sighting descriptions")

# 5. Correlation Analysis
print("\n--- Correlation Analysis ---")

# Time correlations
if all(col in df.columns for col in ['Hour', 'Month', 'Day', 'DayOfWeek']):
    time_features = df[['Hour', 'Month', 'Day', 'DayOfWeek']].copy()
    corr = time_features.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Time Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/time_correlation.png')

# Do certain shapes appear more at specific times?
if all(col in df.columns for col in ['Shape', 'Hour']):
    plt.figure(figsize=(15, 10))
    
    # Get top 5 shapes
    top_shapes = df['Shape'].value_counts().head(5).index.tolist()
    
    # Filter data
    shape_hour_data = df[df['Shape'].isin(top_shapes)]
    
    # Create pivot table
    shape_hour_pivot = pd.crosstab(
        index=shape_hour_data['Hour'],
        columns=shape_hour_data['Shape'],
        normalize='columns'  # Normalize by column to get percentage
    )
    
    # Plot heatmap
    sns.heatmap(shape_hour_pivot, cmap='YlGnBu', annot=False)
    plt.title('UFO Shape Distribution by Hour of Day (Normalized)', fontsize=16)
    plt.xlabel('Shape', fontsize=14)
    plt.ylabel('Hour of Day', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/shape_hour_heatmap.png')
    print("Created heatmap showing relationship between UFO shapes and time of day")

# 6. Geospatial Density Analysis using Plotly
print("\n--- Interactive Geospatial Analysis ---")

if all(col in df.columns for col in ['lat', 'lng']):
    # Create a density map using Plotly
    fig = go.Figure(data=go.Densitymapbox(
        lat=df['lat'],
        lon=df['lng'],
        radius=10,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Density')
    ))
    
    fig.update_layout(
        title='UFO Sightings Density Map',
        mapbox_style="carto-darkmatter",
        mapbox=dict(
            center=dict(lat=df['lat'].mean(), lon=df['lng'].mean()),
            zoom=3
        ),
        width=1000,
        height=800
    )
    
    # Save to HTML file
    fig.write_html('visualizations/density_map_plotly.html')
    print("Created interactive density map using Plotly")

# 7. Summary Statistics
print("\n--- Summary Statistics and Insights ---")

# Number of sightings
print(f"Total number of UFO sightings in dataset: {len(df)}")

# Top countries
country_summary = df['Country'].value_counts().head(5)
print("\nTop 5 countries by number of sightings:")
for country, count in country_summary.items():
    print(f"  {country}: {count} sightings")

# Top shapes
if 'Shape' in df.columns:
    shape_summary = df['Shape'].value_counts().head(5)
    print("\nTop 5 UFO shapes reported:")
    for shape, count in shape_summary.items():
        print(f"  {shape}: {count} sightings")

# Time patterns
if 'Hour' in df.columns:
    night_count = df[df['Hour'].isin(range(18, 24)) | df['Hour'].isin(range(0, 6))].shape[0]
    day_count = df[df['Hour'].isin(range(6, 18))].shape[0]
    night_percentage = (night_count / (night_count + day_count)) * 100
    
    print(f"\nNight vs Day sightings:")
    print(f"  Night (6 PM - 6 AM): {night_count} sightings ({night_percentage:.1f}%)")
    print(f"  Day (6 AM - 6 PM): {day_count} sightings ({100-night_percentage:.1f}%)")

# Final message
print("\nAnalysis complete! All visualizations saved to the 'visualizations' directory.")
print("For interactive visualizations, open the HTML files in a web browser.") 