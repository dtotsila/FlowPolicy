# Flow Matching Policy for Action Chunking

A PyTorch implementation of a Flow Matching policy using a Diffusion Transformer (DiT) backbone. This repository focuses on modeling trajectory generation and robotic action chunking, featuring temporal ensembling and receding horizon deployment strategies.

## Project Structure

* `configs/`: YAML configuration files for experiments (`toy_2d.yaml`, `lasa.yaml`).
* `data/`: Dataset loaders, including toy 2D circles and the LASA handwriting dataset.
* `models/`: Neural network architectures (e.g., `dit.py`).
* `policies/`: Formulations for continuous-time ODE policies (`flow_matcher.py`).
* `scripts/`: Executable scripts for training (`train_toy.py`, `train_lasa.py`), deployment (`deploy_toy.py`, `deploy_lasa.py`), and visualization (`visualize_flow.py`).
* `utils/`: Helper functions for visualization and setting deterministic seeds.

## Setup

Install dependencies:
```bash
pip install torch numpy matplotlib pyyaml pylasadataset
```

## Usage

**Train a policy on the LASA dataset:**
```bash
python scripts/train_lasa.py
```

**Deploy and evaluate using Temporal Ensembling:**
```bash
python scripts/deploy_lasa.py
```

**Visualize the learned Flow Matching vector field:**
```bash
python scripts/visualize_flow.py
```