# scripts/deploy_lasa.py
import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyLasaDataset as lasa

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dit import DiTPolicy
from policies.flow_matcher import FlowMatcher
from utils.seed import set_seed
from utils.normalizer import DictNormalizer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def closed_loop_rollout(policy, normalizer, initial_state, steps, chunk_size, num_inference_steps, k=1):
    """Starts at initial_state, predicts chunk, executes K steps, updates position, repeats."""
    executed_actions = []
    current_state = initial_state

    for _ in range(0, steps, k):
        # 1. Normalize current state using 'state' key
        norm_state = normalizer.normalize('state', current_state)

        # 2. Predict next chunk and denormalize using 'action' key
        norm_delta_chunk = policy.sample(norm_state, chunk_size, action_dim=2, num_steps=num_inference_steps).squeeze(0)
        delta_chunk = normalizer.denormalize('action', norm_delta_chunk).cpu().numpy()
        absolute_chunk = delta_chunk + current_state.cpu().numpy()

        # 3. Take the next k step(s)
        executed_k = absolute_chunk[:k]
        executed_actions.append(executed_k)

        # 4. Update current position to the last step taken
        current_state = torch.tensor(executed_k[-1], dtype=torch.float32).unsqueeze(0).to(initial_state.device)

    return np.concatenate(executed_actions, axis=0)[:steps]

if __name__ == "__main__":
    set_seed(42)
    config = load_config("configs/lasa.yaml")
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

    chunk_size = config['dataset']['chunk_size']
    pattern = config['dataset']['pattern_name']

    # Load Model
    model = DiTPolicy(
        action_dim=config['dataset']['action_dim'],
        state_dim=config['dataset']['state_dim'],
        chunk_size=chunk_size,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads']
    ).to(device)
    checkpoint = torch.load(f"weights/{config['project_name']}.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])

    normalizer = DictNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])

    model.eval()
    policy = FlowMatcher(model).to(device)

    pattern_data = getattr(lasa.DataSet, pattern)

    # Setup subplots for Train and Test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"LASA Deployment: {pattern} (Closed-Loop Rollout)")

    # --- Evaluate Training Demos ---
    print("Evaluating Training Demos...")
    for i, idx in enumerate(config['dataset']['train_indices']):
        demo = pattern_data.demos[idx].pos.T
        start_state = torch.tensor(demo[0], dtype=torch.float32).unsqueeze(0).to(device)

        actions = closed_loop_rollout(
            policy, normalizer, start_state, len(demo),
            chunk_size, config['inference']['num_steps'], k=1
        )

        ax1.plot(demo[:, 0], demo[:, 1], 'g--', label="Ground Truth" if i==0 else "", alpha=0.4)
        ax1.plot(actions[:, 0], actions[:, 1], 'b-', label="Generated" if i==0 else "", alpha=0.8)
        ax1.plot(demo[0, 0], demo[0, 1], 'ro', label="Start" if i==0 else "")

    ax1.set_title("Training Data")
    ax1.axis('equal')
    ax1.legend()
    ax1.grid(True)

    # --- Evaluate Testing Demos ---
    print("Evaluating Testing Demos...")
    for i, idx in enumerate(config['dataset']['test_indices']):
        demo = pattern_data.demos[idx].pos.T
        start_state = torch.tensor(demo[0], dtype=torch.float32).unsqueeze(0).to(device)

        actions = closed_loop_rollout(
            policy, normalizer, start_state, len(demo),
            chunk_size, config['inference']['num_steps'], k=1
        )

        ax2.plot(demo[:, 0], demo[:, 1], 'g--', label="Ground Truth" if i==0 else "", alpha=0.4)
        ax2.plot(actions[:, 0], actions[:, 1], 'b-', label="Generated" if i==0 else "", alpha=0.8)
        ax2.plot(demo[0, 0], demo[0, 1], 'ro', label="Start" if i==0 else "")

    ax2.set_title("Testing Data")
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/lasa_deployment_full.png")
    print("Saved full deployment visualization to results/lasa_deployment_full.png")