# scripts/visualize_flow.py
import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dit import DiTPolicy
from policies.flow_matcher import FlowMatcher

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config("configs/toy_2d.yaml")
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

    chunk_size = config['dataset']['chunk_size']
    action_dim = config['dataset']['action_dim']
    state_dim = config['dataset']['state_dim']

    # Load Model
    model = DiTPolicy(
        action_dim=action_dim, state_dim=state_dim, chunk_size=chunk_size,
        hidden_dim=config['model']['hidden_dim'], num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads']
    ).to(device)
    model.load_state_dict(torch.load(f"weights/{config['project_name']}.pt", map_location=device))
    model.eval()
    policy = FlowMatcher(model).to(device)

    # Setup Inference
    num_samples = 200
    num_steps = 30
    dt = 1.0 / num_steps

    # Fixed starting state (e.g., right side of the circle)
    state = torch.tensor([[1.0, 0.0]]).repeat(num_samples, 1).to(device)

    # Start with Gaussian Noise
    x_t = torch.randn((num_samples, chunk_size, action_dim), device=device)

    # Store history
    history = [x_t.cpu().numpy()]

    print("Solving ODE and tracking flow...")
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.ones((num_samples, 1), device=device) * (i / num_steps)
            v_t = policy.model(x_t, state, t)
            x_t = x_t + v_t * dt
            history.append(x_t.cpu().numpy())

    history = np.array(history) # Shape: [steps+1, samples, chunk_size, 2]

    # Plotting
    plt.figure(figsize=(8, 8))

    # Plot the flow lines for the 1st action in the chunk across all samples
    for i in range(num_samples):
        path = history[:, i, 0, :] # Shape: [steps+1, 2]
        plt.plot(path[:, 0], path[:, 1], color='gray', alpha=0.3, linewidth=1)

    # Mark start and end distributions
    starts = history[0, :, 0, :]
    ends = history[-1, :, 0, :]

    plt.scatter(starts[:, 0], starts[:, 1], c='blue', label='Noise (t=0)', alpha=0.5, s=15)
    plt.scatter(ends[:, 0], ends[:, 1], c='red', label='Target Shape (t=1)', alpha=0.8, s=20)

    plt.title("Learned Flow: Noise to Target Distribution")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/flow_lines.png")
    print("Saved flow visualization to results/flow_lines.png")