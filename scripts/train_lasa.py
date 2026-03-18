import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.lasa_dataset import LasaDataset
from models.dit import DiTPolicy
from policies.flow_matcher import FlowMatcher
from utils.seed import set_seed
from utils.visualization import plot_lasa_trajectories

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    set_seed(42)
    config = load_config("configs/lasa.yaml")
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

    train_indices = config['dataset']['train_indices']
    test_indices = config['dataset']['test_indices']
    pattern = config['dataset']['pattern_name']

    # Load Training Data
    train_dataset = LasaDataset(
        pattern_name=pattern,
        chunk_size=config['dataset']['chunk_size'],
        demo_indices=train_indices
    )
    dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Init Model & Policy
    model = DiTPolicy(
        action_dim=config['dataset']['action_dim'],
        state_dim=config['dataset']['state_dim'],
        chunk_size=config['dataset']['chunk_size'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads']
    ).to(device)
    policy = FlowMatcher(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    # Training Loop
    print(f"Training on LASA Pattern: {pattern}")
    for epoch in range(config['training']['epochs']):
        total_loss = 0
        for state, action_chunk in dataloader:
            state, action_chunk = state.to(device), action_chunk.to(device)
            optimizer.zero_grad()
            loss = policy.compute_loss(action_chunk, state)
            loss.backward()
            # Add this line right before optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == config['training']['epochs'] - 1:
            print(f"Epoch {epoch:03d} | Average Loss: {total_loss / len(dataloader):.4f}")

    # Save Weights
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), f"weights/{config['project_name']}.pth")
    print(f"Saved weights to weights/{config['project_name']}.pth")

    # Save Ground Truth Visualization
    os.makedirs("results", exist_ok=True)
    plot_lasa_trajectories(pattern, train_indices, test_indices, save_path="results/lasa_base_data.png")
    print("Saved ground truth visualization to results/lasa_base_data.png")

if __name__ == "__main__":
    main()