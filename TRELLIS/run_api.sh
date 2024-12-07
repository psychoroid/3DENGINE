#!/bin/bash

# Set environment variables
export ATTN_BACKEND='xformers'
export SPCONV_ALGO='native'

# Start the FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload 