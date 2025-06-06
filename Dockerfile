# Use the official Python image for the base image.
FROM python:3.11-slim
SHELL ["/bin/bash", "-c"]

# Set environment variables to make Python print directly to the terminal and avoid .pyc files.
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Install system dependencies required for package manager and build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    ssh \
    && rm -rf /var/lib/apt/lists/*

# Install pipx.
RUN python3 -m pip install --no-cache-dir pipx \
    && pipx ensurepath

# Add local bin to the path
ENV PATH="${PATH}:/root/.local/bin"


# Install the latest version of uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set the working directory
WORKDIR /workspaces/serena

# Copy required files into the image
COPY pyproject.toml /workspaces/serena/
COPY README.md /workspaces/serena/

# Create virtual environment and install dependencies
RUN uv venv
RUN . .venv/bin/activate
RUN uv pip install --all-extras -r pyproject.toml -e .
ENV PATH="/workspaces/serena/.venv/bin:${PATH}"

# Entrypoint to ensure environment is activated
ENTRYPOINT ["/bin/bash", "-c", "source .venv/bin/activate && $0 $@"]

