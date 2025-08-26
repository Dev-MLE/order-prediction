FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files first (better build cache)
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-interaction --no-ansi

# Copy source code and assets
COPY src/ ./src/
COPY data/ ./data/
COPY .env .env

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app with uvicorn (recommended for ASGI)
CMD ["poetry", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
