# Run 'docker compose up --build' from project root to run the full pipeline
# Use 'docker compose run --entrypoint python pipeline {filename}' to run files separately

# Uses NVIDIA NGC container for GPU-accelerated training
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /workspace/photomedicine_labs

# System libraries for imgaug
RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Installs Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies all project files
COPY . .

# Runs training pipeline: data prep then both model stages
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
