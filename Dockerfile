# Use a Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose the correct Streamlit port
EXPOSE 8501

# Run the app on the correct port
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]