import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dit import DiTPolicy
from policies.flow_matcher import FlowMatcher

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def receding_horizon(policy, initial_state, steps, chunk_size, k, num_inference_steps):
    """Executes K steps, updates state to the current position, then predicts."""
    executed_actions = []
    current_state = initial_state

    for i in range(0, steps, k):
        chunk = policy.sample(current_state, chunk_size, action_dim=2, sampling_steps=num_inference_steps).squeeze(0)
        executed_k = chunk[:k].cpu().numpy()
        executed_actions.append(executed_k)
        current_state = torch.tensor(executed_k[-1:], dtype=torch.float32).to(initial_state.device)

    return np.concatenate(executed_actions, axis=0)[:steps]

def temporal_ensembling(policy, initial_state, steps, chunk_size, num_inference_steps):
    """Predicts every step, averages overlapping chunks, and updates state."""
    executed_actions = []
    all_predicted_chunks = []

    current_state = initial_state
    action_buffer = [[] for _ in range(steps + chunk_size)]

    for t in range(steps):
        chunk = policy.sample(current_state, chunk_size, action_dim=2, sampling_steps=num_inference_steps).squeeze(0).cpu().numpy()
        all_predicted_chunks.append(chunk)

        for offset in range(chunk_size):
            action_buffer[t + offset].append(chunk[offset])

        current_action = np.mean(action_buffer[t], axis=0)
        executed_actions.append(current_action)

        current_state = torch.tensor(current_action, dtype=torch.float32).unsqueeze(0).to(initial_state.device)

    return np.array(executed_actions), np.array(all_predicted_chunks)

def create_ensembling_gif(executed_actions, predicted_chunks, chunk_size, filename="temporal_ensembling.gif"):
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear() # Completely delete everything from the previous step
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f"Temporal Ensembling - Step {frame}")
        ax.grid(True)

        # 1. Get only the ensembles currently active at this step
        start_idx = max(0, frame - chunk_size + 1)
        active_chunks = predicted_chunks[start_idx : frame + 1]

        # 2. Plot ONLY the current active ensembles in low opacity red
        for chunk in active_chunks:
            ax.plot(chunk[:, 0], chunk[:, 1], 'r-', alpha=0.3)

        # 3. Plot the executed path so far
        path_so_far = executed_actions[:frame+1]
        ax.plot(path_so_far[:, 0], path_so_far[:, 1], 'b-', linewidth=2, alpha=1.0)

        # 4. Plot the final ensembled action for this specific step in full opacity
        ax.plot(executed_actions[frame, 0], executed_actions[frame, 1], 'bo', markersize=8)

    # Set blit=False to prevent matplotlib from leaving artifacts of old frames
    ani = animation.FuncAnimation(fig, update, frames=len(executed_actions), blit=False)
    ani.save(filename, writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    # Load configuration
    config = load_config("configs/toy_2d.yaml")

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    chunk_size = config['dataset']['chunk_size']
    action_dim = config['dataset']['action_dim']
    state_dim = config['dataset']['state_dim']
    steps = config['inference']['sampling_steps']
    num_inference_steps = config['inference']['sampling_steps']

    # 1. Load the trained model
    model = DiTPolicy(
        action_dim=action_dim,
        state_dim=state_dim,
        chunk_size=chunk_size,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads']
    ).to(device)

    # Matches the saved weights naming convention from your training script
    weights_path = f"weights/{config['project_name']}.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    policy = FlowMatcher(model).to(device)

    # 2. Start state: Point on the right edge of the circle
    start_state = torch.tensor([[1.0, 0.0]]).to(device)

    print("Simulating Temporal Ensembling...")
    te_actions, te_chunks = temporal_ensembling(
        policy,
        start_state,
        steps=steps,
        chunk_size=chunk_size,
        num_inference_steps=num_inference_steps
    )

    print("Generating GIF...")
    os.makedirs("results", exist_ok=True)
    create_ensembling_gif(
        te_actions,
        te_chunks,
        chunk_size=chunk_size,
        filename="results/temporal_ensembling.gif"
    )