FROM python:3.12-slim-bookworm

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

CMD ["bash"]
