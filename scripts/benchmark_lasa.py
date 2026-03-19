# scripts/eval_batch.py
import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets import LasaDataset
from models.dit import DiTPolicy
from policies.flow_matcher import FlowMatcher
from utils.normalizer import DictNormalizer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_batch(dataloader, policy, normalizer, device, num_inference_steps):
    policy.eval()
    total_l2_error = 0
    total_items = 0

    with torch.no_grad():
        for state, gt_delta_chunk in dataloader:
            state, gt_delta_chunk = state.to(device), gt_delta_chunk.to(device)

            norm_state = normalizer.normalize('state', state)

            # Predict chunk
            norm_pred_chunk = policy.sample(
                norm_state,
                chunk_size=gt_delta_chunk.shape[1],
                action_dim=2,
                sampling_steps=num_inference_steps
            )

            pred_delta_chunk = normalizer.denormalize('action', norm_pred_chunk)

            # L2 distance (Euclidean distance) per timestep
            l2_errors = torch.norm(pred_delta_chunk - gt_delta_chunk, p=2, dim=-1)

            total_l2_error += l2_errors.sum().item()
            total_items += l2_errors.numel()

    return total_l2_error / total_items

if __name__ == "__main__":
    config = load_config("configs/lasa.yaml")
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

    # Load Normalizer
    normalizer = DictNormalizer()
    checkpoint = torch.load(f"weights/{config.get('project_name', 'lasa')}.pt", map_location=device)
    normalizer.load_state_dict(checkpoint["normalizer"])

    # Load Model
    model = DiTPolicy(
        action_dim=config['dataset']['action_dim'],
        state_dim=config['dataset']['state_dim'],
        chunk_size=config['dataset']['chunk_size'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads']
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    policy = FlowMatcher(model).to(device)

    # Setup Datasets
    train_dataset = LasaDataset(
        pattern_name=config['dataset']['pattern_name'],
        chunk_size=config['dataset']['chunk_size'],
        demo_indices=config['dataset']['train_indices'],
        include_velocity=config['dataset'].get('use_velocity', False),
        include_acceleration=config['dataset'].get('use_acceleration', False)
    )

    test_dataset = LasaDataset(
        pattern_name=config['dataset']['pattern_name'],
        chunk_size=config['dataset']['chunk_size'],
        demo_indices=config['dataset']['test_indices'],
        include_velocity=config['dataset'].get('use_velocity', False),
        include_acceleration=config['dataset'].get('use_acceleration', False)
    )

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    sampling_steps = config['inference']['sampling_steps']

    train_error = evaluate_batch(train_loader, policy, normalizer, device, sampling_steps)
    test_error = evaluate_batch(test_loader, policy, normalizer, device, sampling_steps)

    print(f"Batch Evaluation for {config['dataset']['pattern_name']}")
    print(f"Mean Train Position Error: {train_error:.4f}")
    print(f"Mean Test Position Error: {test_error:.4f}")