FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /workspace/photomedicine_labs

# System libraries for imgaug
RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Run training pipeline: data prep then both model stages
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
