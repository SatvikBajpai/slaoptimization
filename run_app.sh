#!/bin/bash
# Script to run the Dark Store Order Processing Simulator
# Usage: ./run_app.sh

echo "Starting Dark Store Order Processing Simulator..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: streamlit is not installed. Please install it using 'pip install streamlit'."
    exit 1
fi

# Run the app
streamlit run streamlit_app.py

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Failed to start the Streamlit app."
    exit 1
fi