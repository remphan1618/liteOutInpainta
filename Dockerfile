# Use official PyTorch image with CUDA 11.8 and cuDNN, suitable for NVIDIA GPUs or CPU fallback
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && \
    apt-get install -y git wget ffmpeg libgl1-mesa-glx libglib2.0-0 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Jupyter Notebook & core Python packages
RUN pip install --upgrade pip && \
    pip install jupyter notebook

# Install HuggingFace diffusers, transformers, and all needed dependencies
RUN pip install diffusers==0.27.2 \
    transformers==4.39.3 \
    accelerate==0.29.3 \
    huggingface_hub==0.22.2 \
    gradio==4.23.0 \
    opencv-python==4.9.0.80 \
    matplotlib==3.8.4 \
    pillow==10.3.0 \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# (Optional) Install xformers for performance if needed
RUN pip install xformers==0.0.25

# Clone your repo
WORKDIR /workspace
RUN git clone https://github.com/remphan1618/liteOutInpainta.git
WORKDIR /workspace/liteOutInpainta

# Expose Jupyter (8888) and Gradio (7860) ports
EXPOSE 8888 7860

# Start Jupyter Notebook server by default (can override CMD in docker run)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
