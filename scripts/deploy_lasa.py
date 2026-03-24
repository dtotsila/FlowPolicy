import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyLasaDataset as lasa
import argparse

# You can remove this block if you adopted the PYTHONPATH or setup.py approach!
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, build_run_name
from data.normalizer import DictNormalizer
from utils.seed import set_seed
from policies.utils.ensembler import BatchedTemporalEnsembler
from policies.flow_matcher import FlowMatcher
from models.dit import DiTPolicy

def plot_streamlines(ax, policy, normalizer, pattern_data, device, chunk_size, num_inference_steps, class_id=None, grid_size=40):
    all_pos = np.concatenate([d.pos.T for d in pattern_data.demos])
    x_min, x_max = all_pos[:, 0].min() - 5, all_pos[:, 0].max() + 5
    y_min, y_max = all_pos[:, 1].min() - 5, all_pos[:, 1].max() + 5

    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    pts = np.c_[X.ravel(), Y.ravel()]

    U = np.zeros(len(pts))
    V = np.zeros(len(pts))
    batch_size = 256

    print(f"Generating streamlines for {ax.get_title()}...")
    with torch.no_grad():
        for i in range(0, len(pts), batch_size):
            p = torch.tensor(pts[i:i+batch_size], dtype=torch.float32).to(device)
            norm_p = normalizer.normalize('state', p)

            # Create batched condition
            cond = torch.full((p.shape[0],), class_id, dtype=torch.long, device=device) if class_id is not None else None

            norm_chunk = policy.sample(norm_p, chunk_size, action_dim=2, sampling_steps=num_inference_steps, condition=cond)
            chunk = normalizer.denormalize('action', norm_chunk).cpu().numpy()

            U[i:i+batch_size] = chunk[:, 0, 0]
            V[i:i+batch_size] = chunk[:, 0, 1]

    ax.streamplot(X, Y, U.reshape(grid_size, grid_size), V.reshape(grid_size, grid_size),
                  color='lightgray', density=1.5, linewidth=0.8, arrowsize=1.0, zorder=0)


def batched_closed_loop_rollout(policy, normalizer, batched_initial_states, steps, chunk_size, num_inference_steps, action_dim, class_id=None, k=1, exp_weight=0.01):
    executed_actions = []
    current_states = batched_initial_states

    ensembler = BatchedTemporalEnsembler(exp_weight=exp_weight)

    # Create batched condition for the rollout
    cond = torch.full((batched_initial_states.shape[0],), class_id, dtype=torch.long, device=batched_initial_states.device) if class_id is not None else None

    for t in range(0, steps, k):
        norm_states = normalizer.normalize('state', current_states)

        # Pass the condition to the sampler
        norm_delta_chunk = policy.sample(norm_states, chunk_size, action_dim=action_dim, sampling_steps=num_inference_steps, condition=cond)
        delta_chunk = normalizer.denormalize('action', norm_delta_chunk).cpu().numpy()

        absolute_chunk = delta_chunk + current_states.cpu().numpy()[:, np.newaxis, :]
        ensembler.update(t, absolute_chunk)

        executed_k = []
        for step_idx in range(t, min(t + k, steps)):
            executed_k.append(ensembler.get_action(step_idx))

        executed_k = np.stack(executed_k, axis=1)
        executed_actions.append(executed_k)
        current_states = torch.tensor(executed_k[:, -1, :], dtype=torch.float32).to(batched_initial_states.device)

    return np.concatenate(executed_actions, axis=1)[:, :steps, :]


def evaluate_split(indices, ax, title, policy, normalizer, pattern_data, config, device, class_id=None):
    print(f"Evaluating {title}...")
    chunk_size = config['dataset']['chunk_size']

    start_states_list = []
    pos_list = []

    for idx in indices:
        demo = pattern_data.demos[idx]
        pos = demo.pos.T
        pos_list.append(pos)

        state_components = [pos[0]]
        if config['dataset'].get('use_velocity', False):
            state_components.append(demo.vel.T[0])
        if config['dataset'].get('use_acceleration', False):
            state_components.append(demo.acc.T[0])

        start_states_list.append(np.concatenate(state_components))

    batched_start_states = torch.tensor(np.array(start_states_list), dtype=torch.float32).to(device)
    steps = len(pos_list[0])

    batched_actions = batched_closed_loop_rollout(
        policy, normalizer, batched_start_states, steps,
        chunk_size, config['inference']['sampling_steps'],
        action_dim=config['dataset']['state_dim'],
        class_id=class_id, # Pass class_id here
        k=config['inference'].get('k_step', 1)
    )

    for i, pos in enumerate(pos_list):
        ax.plot(pos[:, 0], pos[:, 1], 'g--', label="Ground Truth" if i == 0 else "", alpha=0.4, zorder=1)
        ax.plot(batched_actions[i][:, 0], batched_actions[i][:, 1], 'b-', label="Generated" if i == 0 else "", alpha=0.8, zorder=2)
        ax.plot(pos[0, 0], pos[0, 1], 'ro', label="Start" if i == 0 else "", zorder=3)

    ax.set_title(title)
    ax.axis('equal')
    ax.legend()
    ax.grid(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    set_seed(42)
    config = load_config(args.config)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

    chunk_size = config['dataset']['chunk_size']

    # Extract pattern_names list from config
    pattern_names = config['dataset'].get('pattern_names', ["Angle"])
    # Ensure pattern_names is a list
    if isinstance(pattern_names, str):
        pattern_names = [pattern_names]

    run_name = build_run_name(config)

    # Load Model (Now including num_classes)
    model = DiTPolicy(
        action_dim=config['dataset']['action_dim'],
        state_dim=config['dataset']['state_dim'],
        chunk_size=chunk_size,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        num_classes=config['model'].get('num_classes', None)
    ).to(device)

    checkpoint_path = f"weights/{run_name}_best.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find weights at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    normalizer = DictNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])

    model.eval()
    policy = FlowMatcher(model).to(device)

    os.makedirs("results", exist_ok=True)

    # Loop through each pattern and generate a separate plot
    for class_id, pattern in enumerate(pattern_names):
        print(f"\n--- Processing Pattern: {pattern} (Class ID: {class_id}) ---")
        pattern_data = getattr(lasa.DataSet, pattern)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"LASA Deployment: {pattern} | {run_name}")
        ax1.set_title("Training Data")
        ax2.set_title("Testing Data")

        if config['dataset']['state_dim'] == 2:
            plot_streamlines(ax1, policy, normalizer, pattern_data, device, chunk_size, config['inference']['sampling_steps'], class_id=class_id)
            plot_streamlines(ax2, policy, normalizer, pattern_data, device, chunk_size, config['inference']['sampling_steps'], class_id=class_id)

        evaluate_split(config['dataset']['train_indices'], ax1, "Training Data", policy, normalizer, pattern_data, config, device, class_id=class_id)
        evaluate_split(config['dataset']['test_indices'], ax2, "Testing Data", policy, normalizer, pattern_data, config, device, class_id=class_id)

        save_path = f"results/{run_name}_{pattern}_deployment.png"
        plt.savefig(save_path)
        plt.close(fig) # Close the figure to save memory
        print(f"Saved deployment visualization to {save_path}")