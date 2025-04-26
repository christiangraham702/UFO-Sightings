#!/bin/bash

# Install dependencies if needed
if [ "$1" == "--install" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the Streamlit app
echo "Starting UFO Sightings Analysis Dashboard..."
streamlit run app.py 