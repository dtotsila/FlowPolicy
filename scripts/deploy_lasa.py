import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyLasaDataset as lasa

# You can remove this block if you adopted the PYTHONPATH or setup.py approach!
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.config import load_config, build_run_name
from utils.normalizer import DictNormalizer
from utils.seed import set_seed
from utils.ensembler import BatchedTemporalEnsembler
from policies.flow_matcher import FlowMatcher
from models.dit import DiTPolicy

def plot_streamlines(ax, policy, normalizer, pattern_data, device, chunk_size, num_inference_steps, grid_size=40):
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

            norm_chunk = policy.sample(norm_p, chunk_size, action_dim=2, sampling_steps=num_inference_steps)
            chunk = normalizer.denormalize('action', norm_chunk).cpu().numpy()

            U[i:i+batch_size] = chunk[:, 0, 0]
            V[i:i+batch_size] = chunk[:, 0, 1]

    ax.streamplot(X, Y, U.reshape(grid_size, grid_size), V.reshape(grid_size, grid_size),
                  color='lightgray', density=1.5, linewidth=0.8, arrowsize=1.0, zorder=0)


def batched_closed_loop_rollout(policy, normalizer, batched_initial_states, steps, chunk_size, num_inference_steps, action_dim, k=1, exp_weight=0.01):
    executed_actions = []
    current_states = batched_initial_states

    ensembler = BatchedTemporalEnsembler(exp_weight=exp_weight)

    for t in range(0, steps, k):
        norm_states = normalizer.normalize('state', current_states)

        # Predict the full delta state (pos + vel + acc)
        norm_delta_chunk = policy.sample(norm_states, chunk_size, action_dim=action_dim, sampling_steps=num_inference_steps)
        delta_chunk = normalizer.denormalize('action', norm_delta_chunk).cpu().numpy()

        # absolute_chunk is now the full predicted state (pos, vel, acc)
        absolute_chunk = delta_chunk + current_states.cpu().numpy()[:, np.newaxis, :]

        ensembler.update(t, absolute_chunk)

        executed_k = []
        for step_idx in range(t, min(t + k, steps)):
            executed_k.append(ensembler.get_action(step_idx))

        executed_k = np.stack(executed_k, axis=1) # (B, k, action_dim)
        executed_actions.append(executed_k)

        # Update the state directly from the ensembled model output
        current_states = torch.tensor(executed_k[:, -1, :], dtype=torch.float32).to(batched_initial_states.device)

    return np.concatenate(executed_actions, axis=1)[:, :steps, :]


def evaluate_split(indices, ax, title, policy, normalizer, pattern_data, config, device):
    """Helper to evaluate and plot a specific split of the dataset using batched inference."""
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

    # Create batch: (B, state_dim)
    batched_start_states = torch.tensor(np.array(start_states_list), dtype=torch.float32).to(device)
    steps = len(pos_list[0]) # LASA sequences are all length 1000

    # Single batched rollout
    batched_actions = batched_closed_loop_rollout(
        policy, normalizer, batched_start_states, steps,
        chunk_size, config['inference']['sampling_steps'],
        action_dim=config['dataset']['state_dim'], # Pass the full state dimension
        k=config['inference'].get('k_step', 1)
    )

    # Plot results
    for i, pos in enumerate(pos_list):
        ax.plot(pos[:, 0], pos[:, 1], 'g--', label="Ground Truth" if i == 0 else "", alpha=0.4, zorder=1)
        ax.plot(batched_actions[i][:, 0], batched_actions[i][:, 1], 'b-', label="Generated" if i == 0 else "", alpha=0.8, zorder=2)
        ax.plot(pos[0, 0], pos[0, 1], 'ro', label="Start" if i == 0 else "", zorder=3)

    ax.set_title(title)
    ax.axis('equal')
    ax.legend()
    ax.grid(True)


if __name__ == "__main__":
    set_seed(42)
    config = load_config("configs/lasa.yaml")
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

    chunk_size = config['dataset']['chunk_size']
    pattern = config['dataset']['pattern_name']

    # 1. Reconstruct Dynamic Run Name
    dataset_name = config['dataset'].get('name', 'lasa').lower()
    lr = config['training']['lr']
    hidden = config['model']['hidden_dim']

    run_name = build_run_name(config)
    # 2. Load Model
    model = DiTPolicy(
        action_dim=config['dataset']['action_dim'],
        state_dim=config['dataset']['state_dim'],
        chunk_size=chunk_size,
        hidden_dim=hidden,
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads']
    ).to(device)

    checkpoint_path = f"weights/{run_name}_best.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find weights at {checkpoint_path}. Did the model finish training?")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    normalizer = DictNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])

    model.eval()
    policy = FlowMatcher(model).to(device)
    pattern_data = getattr(lasa.DataSet, pattern)

    # 3. Setup Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"LASA Deployment: {pattern} | {run_name}")
    ax1.set_title("Training Data")
    ax2.set_title("Testing Data")

    # Plot streamlines FIRST
    if config['dataset']['state_dim'] == 2:
        plot_streamlines(ax1, policy, normalizer, pattern_data, device, chunk_size, config['inference']['sampling_steps'])
        plot_streamlines(ax2, policy, normalizer, pattern_data, device, chunk_size, config['inference']['sampling_steps'])

    # 4. Run Evaluations (Batched!)
    evaluate_split(config['dataset']['train_indices'], ax1, "Training Data", policy, normalizer, pattern_data, config, device)
    evaluate_split(config['dataset']['test_indices'], ax2, "Testing Data", policy, normalizer, pattern_data, config, device)

    # 5. Save Results
    os.makedirs("results", exist_ok=True)
    save_path = f"results/{run_name}_deployment.png"
    plt.savefig(save_path)
    print(f"Saved full deployment visualization to {save_path}")