#!/bin/bash

# Change to the project root (same folder as this file)
cd "$(dirname "$0")"

# Activate the smartgraph conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smartgraph

# Open the browser 1 second after the server starts
sleep 1 && open http://localhost:8000 &

# Start the FastAPI server (serves API + frontend)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
