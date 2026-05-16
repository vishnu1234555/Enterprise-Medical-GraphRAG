# Use official lightweight Python image
FROM python:3.11-slim

# Expose NVIDIA GPU variables to enable CUDA within the container
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install necessary system build tools for PyTorch / Sentence Transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
# Upgrade pip to avoid build errors, then install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -v -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]