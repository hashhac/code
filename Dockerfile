# Use official PyTorch image as base (updated tag)
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
# 1. torchcodec (for PyTorch 2.9.0, use torchcodec 0.8)
# 2. torchaudio, soundfile, librosa (audio processing)
# 3. encodec (main.py requires it)
# 4. Upgrade pip and install requirements.txt if present
RUN pip install --upgrade pip \
    && pip install torchcodec==0.8 torchaudio==2.4.0 soundfile librosa encodec hydra-core omegaconf openai-whisper pystoi jiwer inflect scipy pandas matplotlib seaborn tqdm


# If requirements.txt exists, install from it (optional, for user extensibility)
COPY requiremets.txt ./
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Set working directory
WORKDIR /app

# Copy your code
COPY . .

# Set environment variables
ENV TORCHAUDIO_USE_TORCHCODEC=1
ENV PYTHONUNBUFFERED=1

# Command to run training
CMD ["python", "main.py"]