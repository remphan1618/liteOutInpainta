FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && \
    apt-get install -y git wget ffmpeg libgl1-mesa-glx libglib2.0-0 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Jupyter Notebook & core Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir jupyter notebook

# Install HuggingFace diffusers, transformers, and all needed dependencies (using --no-cache-dir for space)
RUN pip install --no-cache-dir diffusers==0.27.2 \
    transformers==4.39.3 \
    accelerate==0.29.3 \
    huggingface_hub==0.22.2 \
    gradio==4.23.0 \
    opencv-python==4.9.0.80 \
    matplotlib==3.8.4 \
    pillow==10.3.0 \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# xformers is not installed here to save space, but can be installed in the notebook as needed.

# Clone your repo
WORKDIR /workspace
RUN git clone https://github.com/remphan1618/liteOutInpainta.git
WORKDIR /workspace/liteOutInpainta

# Copy the notebook into the image
COPY setup_and_run.ipynb /workspace/liteOutInpainta/

# Expose Jupyter (8888) and Gradio (7860) ports
EXPOSE 8888 7860

# Start Jupyter Notebook server by default
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
