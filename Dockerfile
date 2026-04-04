FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port (HuggingFace Spaces uses 7860)
EXPOSE 7860

# Run Uvicorn server
CMD ["uvicorn", "app.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
