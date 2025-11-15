FROM mcr.microsoft.com/devcontainers/python:3.10-bullseye

# Install system dependencies commonly needed for ML workflows.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*
