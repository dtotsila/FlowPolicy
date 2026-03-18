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

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def temporal_ensembling(policy, initial_state, steps, chunk_size, num_inference_steps):
    executed_actions = []
    current_state = initial_state
    action_buffer = [[] for _ in range(steps + chunk_size)]

    for t in range(steps):
        chunk = policy.sample(current_state, chunk_size, action_dim=2, num_steps=num_inference_steps).squeeze(0).cpu().numpy()
        for offset in range(chunk_size):
            action_buffer[t + offset].append(chunk[offset])

        current_action = np.mean(action_buffer[t], axis=0)
        executed_actions.append(current_action)
        current_state = torch.tensor(current_action, dtype=torch.float32).unsqueeze(0).to(initial_state.device)

    return np.array(executed_actions)

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
    model.load_state_dict(torch.load(f"weights/{config['project_name']}.pth", map_location=device))
    model.eval()
    policy = FlowMatcher(model).to(device)

    pattern_data = getattr(lasa.DataSet, pattern)

    # Setup subplots for Train and Test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"LASA Deployment: {pattern}")

    # --- Evaluate Training Demos ---
    print("Evaluating Training Demos...")
    for i, idx in enumerate(config['dataset']['train_indices']):
        demo = pattern_data.demos[idx].pos.T
        start_state = torch.tensor(demo[0], dtype=torch.float32).unsqueeze(0).to(device)
        te_actions = temporal_ensembling(policy, start_state, len(demo), chunk_size, config['inference']['num_steps'])

        ax1.plot(demo[:, 0], demo[:, 1], 'g--', label="Ground Truth" if i==0 else "", alpha=0.4)
        ax1.plot(te_actions[:, 0], te_actions[:, 1], 'b-', label="Generated" if i==0 else "", alpha=0.8)
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
        te_actions = temporal_ensembling(policy, start_state, len(demo), chunk_size, config['inference']['num_steps'])

        ax2.plot(demo[:, 0], demo[:, 1], 'g--', label="Ground Truth" if i==0 else "", alpha=0.4)
        ax2.plot(te_actions[:, 0], te_actions[:, 1], 'b-', label="Generated" if i==0 else "", alpha=0.8)
        ax2.plot(demo[0, 0], demo[0, 1], 'ro', label="Start" if i==0 else "")

    ax2.set_title("Testing Data")
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/lasa_deployment_full.png")
    print("Saved full deployment visualization to results/lasa_deployment_full.png")