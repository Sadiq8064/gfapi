FROM python:3.11-slim

# Prevent Python from buffering stdout (useful for logging)
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Expose FastAPI PORT
EXPOSE 8000

# Start API
CMD ["uvicorn", "gfsapi:app", "--host", "0.0.0.0", "--port", "8000"]
