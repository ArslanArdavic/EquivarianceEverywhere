# Step 1: Base image (CUDA 11.8, Ubuntu 22.04)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Step 2: Minimal OS deps + env
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python-is-python3 \
    python3-dev build-essential \
    git curl ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Step 3: PyTorch 2.3.0 (CUDA 11.8)
RUN python -m pip install --upgrade pip && \
    pip install "torch==2.3.0" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118

# Step 4: PyG compiled extensions for torch 2.3 + cu118
RUN pip install \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

# Step 5: Easy packages via requirements.txt
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Option A: Minimal pass-through entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
