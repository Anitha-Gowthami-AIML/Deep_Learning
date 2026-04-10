#!/bin/bash
# Run script for Spotify Hit Prediction App

echo "🎵 Starting Spotify Hit Prediction App..."
echo "Installing dependencies..."

pip install -r requirements.txt

echo "🚀 Launching Streamlit app..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0