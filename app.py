import streamlit as st
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
from streamlit_folium import folium_static
import os
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="UFO Sightings Analysis Dashboard",
    page_icon="👽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .stApp {
        background-color: #0f172a;
    }
    h1, h2 {
        color: #38bdf8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        color: #38bdf8;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: 1px solid #38bdf8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #38bdf8;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("UFO Sightings Analysis Dashboard")
st.markdown("### Exploring patterns in reported UFO sightings data")

# Sidebar with dataset info and controls
with st.sidebar:
    st.title("About")
    st.markdown("""
    This dashboard presents analysis of UFO sightings data, showing temporal patterns, 
    geographical distributions, and characteristics of reported sightings.
    
    The analysis is purely data-driven and makes no claims about the authenticity of the reports.
    """)
    
    st.subheader("Dataset")
    st.markdown("This analysis uses a dataset of UFO sightings containing location, time, and description details.")
    
    # Add a download sample button
    st.subheader("Sample Data")
    
    # Function to create sample data for download
    def get_sample_data():
        # Create a sample dataframe if needed
        sample_data = pd.DataFrame({
            'Date / Time': ['10/10/1988 21:00', '7/4/2001 18:30', '6/15/2010 22:45', '1/1/2015 00:15', '8/20/2018 14:00'],
            'City': ['Phoenix', 'New York', 'Los Angeles', 'Chicago', 'Houston'],
            'State': ['AZ', 'NY', 'CA', 'IL', 'TX'],
            'Country': ['USA', 'USA', 'USA', 'USA', 'USA'],
            'Shape': ['triangle', 'light', 'sphere', 'cylinder', 'disk'],
            'Summary': [
                'Three triangular objects hovering silently in the night sky.',
                'Bright lights moving in formation above the city skyline.',
                'Spherical object observed moving erratically before disappearing.',
                'Cylinder-shaped craft with flashing lights on New Year\'s morning.',
                'Disk-shaped object during daytime, hovering then accelerating rapidly.'
            ],
            'lat': [33.4484, 40.7128, 34.0522, 41.8781, 29.7604],
            'lng': [-112.0740, -74.0060, -118.2437, -87.6298, -95.3698]
        })
        return sample_data
    
    # Create a download button for the sample data
    sample_df = get_sample_data()
    csv = sample_df.to_csv(index=False)
    st.download_button(
        label="Download Sample Data (CSV)",
        data=csv,
        file_name="sample_ufo_data.csv",
        mime="text/csv"
    )

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('notebook/data/UFOs_coord.csv')
        print(f"Successfully loaded dataset with {df.shape[0]} records.")
        return data_preprocessing(df)
    except Exception as e:
        try:
            df = pd.read_excel('notebook/data/UFOs_coord.xlsx')
            print(f"Successfully loaded Excel dataset with {df.shape[0]} records.")
            return data_preprocessing(df)
        except Exception as e:
            st.warning(f"Error loading data files: {e}")
            
            # Try to find any UFO CSV files in the current directory
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'ufo' in f.lower()]
            if csv_files:
                try:
                    df = pd.read_csv(csv_files[0])
                    st.success(f"Found and loaded alternative dataset: {csv_files[0]}")
                    return data_preprocessing(df)
                except Exception as e:
                    st.error(f"Error loading alternative dataset: {e}")
            
            # If all else fails, use the sample data
            st.info("Using sample data for demonstration. Download the sample and upload below for a demo.")
            
            # File uploader for user to upload their own data
            uploaded_file = st.file_uploader("Upload UFO sightings CSV file", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    return data_preprocessing(df)
                except Exception as e:
                    st.error(f"Error loading uploaded file: {e}")
            
            # Use sample data as fallback
            df = get_sample_data()
            return data_preprocessing(df)

def data_preprocessing(df):
    """Process the dataframe to extract required features"""
    # Extract year from date
    if 'Date / Time' in df.columns:
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
    
    return df

# Load the data
df = load_data()

if df is not None:
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Geographic Analysis", 
        "Temporal Patterns", 
        "UFO Shapes", 
        "Text Analysis",
        "Summary Statistics"
    ])
    
    with tab1:
        st.header("Geographic Distribution of UFO Sightings")
        
        # Create two columns for the maps
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Interactive UFO Sightings Map")
            
            # Create a folium map
            m = folium.Map(
                location=[df['lat'].mean(), df['lng'].mean()], 
                zoom_start=4, 
                tiles='CartoDB dark_matter'
            )
            
            # Add a heatmap layer
            heat_data = [[row['lat'], row['lng']] for _, row in df.iterrows()]
            HeatMap(heat_data, radius=8, blur=10).add_to(m)
            
            # Add a marker cluster for individual sightings
            marker_cluster = MarkerCluster().add_to(m)
            
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
                    icon=folium.Icon(color='green', icon='info-sign')
                ).add_to(marker_cluster)
            
            # Display the map
            folium_static(m, width=600)
        
        with col2:
            st.subheader("UFO Sightings Density Map")
            
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
                mapbox_style="carto-darkmatter",
                mapbox=dict(
                    center=dict(lat=df['lat'].mean(), lon=df['lng'].mean()),
                    zoom=3
                ),
                height=600,
                margin={"r":0,"t":0,"l":0,"b":0}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Country and US state distribution
        st.subheader("Distribution by Country and US States")
        col1, col2 = st.columns(2)
        
        with col1:
            country_counts = df['Country'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            country_counts.plot(kind='bar', ax=ax)
            plt.title('UFO Sightings by Country', fontsize=16)
            plt.xlabel('Country', fontsize=14)
            plt.ylabel('Number of Sightings', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown(f"**Most UFO sightings reported in:** {country_counts.index[0]} with {country_counts.values[0]} sightings")
        
        with col2:
            if 'State' in df.columns:
                usa_data = df[df['Country'] == 'USA']
                if not usa_data.empty:
                    top_states = usa_data['State'].value_counts().head(15)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_states.plot(kind='bar', ax=ax)
                    plt.title('Top 15 US States by UFO Sightings', fontsize=16)
                    plt.xlabel('State', fontsize=14)
                    plt.ylabel('Number of Sightings', fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown(f"**US State with most UFO sightings:** {top_states.index[0]} with {top_states.values[0]} sightings")
    
    with tab2:        
        # Monthly and hourly patterns
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Month' in df.columns:
                st.subheader("UFO Sightings by Month")
                month_order = range(1, 13)
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_counts = df['Month'].value_counts().reindex(month_order)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=month_names, y=monthly_counts.values, ax=ax)
                plt.title('UFO Sightings by Month', fontsize=16)
                plt.xlabel('Month', fontsize=14)
                plt.ylabel('Number of Sightings', fontsize=14)
                
                # Add value labels to the bars
                for i, count in enumerate(monthly_counts.values):
                    ax.text(i, count + 5, str(count), ha='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown(f"**Month with most UFO sightings:** {month_names[monthly_counts.idxmax()-1]} with {monthly_counts.max()} sightings")
        
        with col2:
            if 'Hour' in df.columns:
                st.subheader("UFO Sightings by Hour of Day")
                hourly_counts = df['Hour'].value_counts().sort_index()
                
                # Create custom colors for day/night visualization
                colors = ['#3a6186' if i >= 18 or i <= 5 else '#89253e' for i in hourly_counts.index]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette=colors, ax=ax)
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
                st.pyplot(fig)
                
                night_percentage = df['is_night'].mean() * 100
                st.markdown(f"**Percentage of sightings occurring at night:** {night_percentage:.2f}%")
        
        # Day of week
        if 'DayOfWeek' in df.columns:
            st.subheader("UFO Sightings by Day of Week")
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df['DayOfWeek'].value_counts().reindex(range(7))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=day_names, y=day_counts.values, ax=ax)
            plt.title('UFO Sightings by Day of Week', fontsize=16)
            plt.xlabel('Day', fontsize=14)
            plt.ylabel('Number of Sightings', fontsize=14)
            plt.xticks(rotation=45)
            
            for i, count in enumerate(day_counts.values):
                ax.text(i, count + 5, str(count), ha='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown(f"**Day with most UFO sightings:** {day_names[day_counts.idxmax()]} with {day_counts.max()} sightings")
        
        # Shape-Hour correlation
        if all(col in df.columns for col in ['Shape', 'Hour']):
            st.subheader("UFO Shape Distribution by Hour of Day")
            
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
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(shape_hour_pivot, cmap='YlGnBu', annot=False, ax=ax)
            plt.title('UFO Shape Distribution by Hour of Day (Normalized)', fontsize=16)
            plt.xlabel('Shape', fontsize=14)
            plt.ylabel('Hour of Day', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab3:
        st.header("UFO Shape Analysis")
        
        if 'Shape' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Reported UFO Shapes")
                shape_counts = df['Shape'].value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x=shape_counts.values, y=shape_counts.index, ax=ax)
                plt.title('Top 10 Reported UFO Shapes', fontsize=16)
                plt.xlabel('Number of Sightings', fontsize=14)
                plt.ylabel('Shape', fontsize=14)
                
                # Add value labels
                for i, count in enumerate(shape_counts.values):
                    ax.text(count + 10, i, str(count), va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown(f"**Most common UFO shape:** {shape_counts.index[0]} with {shape_counts.values[0]} sightings")
            
            with col2:
                # Check if shape prevalence has changed over time
                if 'Year' in df.columns:
                    st.subheader("UFO Shape Trends Over Time")
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
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for shape in top_shapes:
                        if shape in shape_by_year.columns:
                            plt.plot(shape_by_year.index, shape_by_year[shape], marker='o', linewidth=2, label=shape)
                    
                    plt.title('UFO Shape Trends Over Time (Top 5 Shapes)', fontsize=16)
                    plt.xlabel('Year', fontsize=14)
                    plt.ylabel('Number of Sightings', fontsize=14)
                    plt.legend(title='Shape')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
    
    with tab4:
        st.header("Text Analysis of UFO Sighting Descriptions")
        
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
            
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Most Common Words in UFO Sighting Descriptions', fontsize=16)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display some example summaries
            st.subheader("Sample UFO Sighting Descriptions")
            sample_summaries = df['Summary'].dropna().sample(5, random_state=42)
            
            for i, summary in enumerate(sample_summaries):
                st.markdown(f"**Sighting {i+1}:** {summary}")
    
    with tab5:
        st.header("Summary Statistics and Insights")
        
        # Number of sightings
        st.metric("Total UFO Sightings", f"{len(df):,}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top countries
            st.subheader("Top Countries by Number of Sightings")
            country_summary = df['Country'].value_counts().head(5)
            
            country_data = pd.DataFrame({
                'Country': country_summary.index,
                'Sightings': country_summary.values
            })
            
            st.dataframe(country_data, hide_index=True, use_container_width=True)
            
            # Time patterns
            if 'Hour' in df.columns:
                st.subheader("Night vs Day Sightings")
                night_count = df[df['Hour'].isin(range(18, 24)) | df['Hour'].isin(range(0, 6))].shape[0]
                day_count = df[df['Hour'].isin(range(6, 18))].shape[0]
                night_percentage = (night_count / (night_count + day_count)) * 100
                
                time_data = pd.DataFrame({
                    'Time Period': ['Night (6 PM - 6 AM)', 'Day (6 AM - 6 PM)'],
                    'Sightings': [night_count, day_count],
                    'Percentage': [f"{night_percentage:.1f}%", f"{100-night_percentage:.1f}%"]
                })
                
                st.dataframe(time_data, hide_index=True, use_container_width=True)
        
        with col2:
            # Top shapes
            if 'Shape' in df.columns:
                st.subheader("Top UFO Shapes Reported")
                shape_summary = df['Shape'].value_counts().head(5)
                
                shape_data = pd.DataFrame({
                    'Shape': shape_summary.index,
                    'Sightings': shape_summary.values
                })
                
                st.dataframe(shape_data, hide_index=True, use_container_width=True)
                
                # Display a simple pie chart of shapes
                fig, ax = plt.subplots(figsize=(10, 6))
                shape_summary.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, cmap='viridis')
                plt.title('Distribution of Top 5 UFO Shapes', fontsize=16)
                plt.ylabel('')
                plt.tight_layout()
                st.pyplot(fig)
else:
    st.error("Failed to load the dataset. Please check the data files in the notebook/data directory.")

# Footer
st.markdown("---")
st.markdown("UFO Sightings Analysis Dashboard - Data analysis without claims about authenticity") 