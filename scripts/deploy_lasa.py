import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyLasaDataset as lasa
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, build_run_name, beautify_run_name
from dataset.normalizer import DictNormalizer
from utils.seed import set_seed
from policies.utils.ensembler import BatchedTemporalEnsembler
from policies.flow_matcher import FlowMatcher
from models.dit import DiTPolicy
from utils.visualization import render_trajectory_to_image, plot_streamlines


def batched_closed_loop_rollout(policy, normalizer, batched_initial_states, steps, chunk_size, num_inference_steps, class_id=None, image=None, k=1, exp_weight=0.01):
    executed_actions = []
    current_states = batched_initial_states
    batch_size = batched_initial_states.shape[0]

    ensembler = BatchedTemporalEnsembler(exp_weight=exp_weight)
    cond = torch.full((batch_size,), class_id, dtype=torch.long, device=batched_initial_states.device) if class_id is not None else None
    batched_image = image.expand(batch_size, -1, -1, -1) if image is not None else None

    for t in range(0, steps, k):
        norm_states = normalizer.normalize('state', current_states)

        norm_delta_chunk = policy.sample(norm_states, chunk_size, sampling_steps=num_inference_steps, condition=cond, image=batched_image)
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


def evaluate_split(indices, ax, title, policy, normalizer, pattern_data, config, device, class_id=None, image=None):
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
        class_id=class_id,
        image=image,
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
    use_vision = config['dataset'].get('use_vision', False)

    pattern_names = config['dataset'].get('pattern_names', ["Angle"])
    if isinstance(pattern_names, str):
        pattern_names = [pattern_names]

    run_name = build_run_name(config)

    model = DiTPolicy(
        action_dim=config['dataset']['action_dim'],
        state_dim=config['dataset']['state_dim'],
        chunk_size=chunk_size,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        num_classes=config['model'].get('num_classes', None),
        use_vision=config['dataset'].get('use_vision', False)
    ).to(device)


    dataset_name = config["dataset"].get("name", "lasa").lower()
    clean_run_name = beautify_run_name(run_name)
    checkpoint_path = os.path.join("weights", dataset_name, clean_run_name, "final.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find weights at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    normalizer = DictNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])

    model.eval()
    policy = FlowMatcher(model).to(device)

    os.makedirs("results", exist_ok=True)

    for class_id, pattern in enumerate(pattern_names):
        print(f"\n--- Processing Pattern: {pattern} (Class ID: {class_id}) ---")
        pattern_data = getattr(lasa.DataSet, pattern)

        img_tensor = None
        if use_vision:
            raw_img = render_trajectory_to_image(pattern_data.demos[0].pos.T, size=(84, 84))
            img_tensor = torch.tensor(raw_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"LASA Deployment: {pattern} | {clean_run_name}") # using clean_run_name here too
        ax1.set_title("Training Data")
        ax2.set_title("Testing Data")

        if config['dataset']['state_dim'] == 2:
            plot_streamlines(ax1, policy, normalizer, pattern_data, device, chunk_size, config['inference']['sampling_steps'], class_id=class_id, image=img_tensor)
            plot_streamlines(ax2, policy, normalizer, pattern_data, device, chunk_size, config['inference']['sampling_steps'], class_id=class_id, image=img_tensor)

        evaluate_split(config['dataset']['train_indices'], ax1, "Training Data", policy, normalizer, pattern_data, config, device, class_id=class_id, image=img_tensor)
        evaluate_split(config['dataset']['test_indices'], ax2, "Testing Data", policy, normalizer, pattern_data, config, device, class_id=class_id, image=img_tensor)

        save_path = f"results/{clean_run_name}_{pattern}_deployment.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved deployment visualization to {save_path}")