FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        build-essential \
        ca-certificates \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        libpq-dev \
        libffi-dev \
        openssh-client \
        pkg-config \
        libssl-dev \
        && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# Install Miniforge
ENV CONDA_DIR=/opt/miniforge
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN conda init bash && conda config --set auto_activate_base false

# Install PyTorch for CUDA 12.4 and core dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
        "transformers>=4.38" \
        "hydra-core>=1.3" \
        "optuna>=3.5" \
        "mlflow[extras]>=2.11" \
        "pandas>=2.0" \
        "numpy>=1.24" \
        "scikit-learn>=1.3" \
        "tqdm>=4.65" \
        "pyyaml>=6.0" \
        "dvc[postgresql]>=3.0" \
        "psycopg2-binary>=2.9" \
        "peft>=0.11"

WORKDIR /workspace

# Keep container running for dev container
CMD ["sleep", "infinity"]
