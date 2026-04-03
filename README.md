# Flow Matching Policy for Action Chunking

A PyTorch implementation of a Flow Matching policy using a Diffusion Transformer (DiT) backbone. This repository focuses on modeling trajectory generation and robotic action chunking, featuring temporal ensembling, receding horizon deployment, and optional vision-based conditioning.

## Project Structure

* `configs/`: YAML configuration files for experiments (e.g., `lasa.yaml`).
* `data/`: Dataset loaders, HDF5 parsers, and state/action normalizers.
* `models/`: Neural network architectures, including the DiT backbone and vision encoders.
* `policies/`: Formulations for continuous-time ODE policies (`flow_matcher.py`) and temporal ensembling.
* `scripts/`: Executable scripts for data processing (`create_hdf5.py`), training (`train.py`), and deployment (`deploy_lasa.py`).
* `utils/`: Helper functions for configuration parsing, visualization, and deterministic seeding.
* `weights/`: Structured output directory for model checkpoints and periodic saves.

## Setup

1. **Environment Variables:**
   Create your local `.env` file from the provided example:
   ```bash
   cp .env.example .env
   ```

2. **Docker Build:**
   Build the container using the provided Makefile to automatically handle user/group IDs:
   ```bash
   make build
   ```

## Data Preparation !!!WIP!!!

Before training, generate the Robomimic-compatible HDF5 dataset (which includes synthesized images for vision conditioning):
```bash
docker compose run --rm flow_matching python scripts/create_hdf5.py --output data/lasa_vision.hdf5
```

## Usage

Execution is managed via the `Makefile` to ensure seamless Docker integration. You can specify a config file using the `CONFIG` variable (defaults to `configs/lasa.yaml`).

**Interactive Training:**
```bash
make train CONFIG=configs/lasa.yaml
```

**Background Training (Detached):**
```bash
make train-bg CONFIG=configs/lasa.yaml
# View live output:
make logs
# Stop the background process:
make stop-train
```

**Deploy and Evaluate:**
Runs a closed-loop rollout using Temporal Ensembling and generates trajectory visualizations in the `results/` folder.
```bash
make deploy CONFIG=configs/lasa.yaml
```

**Cleanup:**
* Remove generated plots: `make clean-results`
* Delete all saved checkpoints: `make clean-weights`
