# Start with a CUDA-enabled PyTorch image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://huggingface.co/spaces/Red1618/Lightning-Painter-Multitool .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir xformers==0.0.20 peft controlnet_aux

# Set environment variables
ENV BASE_MODEL_PATH=/app/models/stabilityai/stable-diffusion-xl-base-1.0
ENV CONTROLNET_MODELS_PATH=/app/models/controlnet
ENV XFORMERS_MEMORY_EFFICIENT_ATTENTION=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Download models (this part takes a while, grab a fucking beer)
RUN python -c "from huggingface_hub import snapshot_download; import os; \
    os.makedirs('models', exist_ok=True); \
    snapshot_download(repo_id='stabilityai/stable-diffusion-xl-base-1.0', cache_dir='models'); \
    os.makedirs('models/controlnet', exist_ok=True); \
    controlnet_models = ['diffusers/controlnet-canny-sdxl-1.0', 'diffusers/controlnet-depth-sdxl-1.0', 'diffusers/sd-controlnet-openpose', 'diffusers/sd-controlnet-scribble']; \
    [snapshot_download(repo_id=model, cache_dir='models/controlnet') for model in controlnet_models]"

# Expose port for Gradio
EXPOSE 7860

# Start the web UI when container runs
CMD ["python", "app.py", "--listen", "0.0.0.0", "--port", "7860"]
