#!/bin/bash
# Quick start script for Trade Performance Tracker

echo "Starting Trade Performance Tracker..."
echo "=================================="
echo ""
echo "The dashboard will open in your browser at http://localhost:8502"
echo "Press Ctrl+C to stop the server"
echo ""

# Run streamlit from the interface folder
streamlit run interface/sports_dashboard.py --server.headless true --server.port 8502
