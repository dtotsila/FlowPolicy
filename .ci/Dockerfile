# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS base

ARG USER_ID=1000
ARG GROUP_ID=1000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a user with your UID/GID
RUN groupadd -g ${GROUP_ID} go && \
    useradd -l -u ${USER_ID} -g go -m go

# Set the working directory and ownership
WORKDIR /workspace
RUN chown -R ${USER_ID}:${GROUP_ID} /workspace

# Ensure the new user has a bashrc with color support
RUN cp /etc/skel/.bashrc /home/go/.bashrc && \
    sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /home/go/.bashrc


# Switch to the new user
USER go

# Copy files (ownership is preserved via --chown)
COPY --chown=${USER_ID}:${GROUP_ID} requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=${USER_ID}:${GROUP_ID} . .

FROM base AS dev
CMD ["/bin/bash"]
