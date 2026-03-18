import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.toy_dataset import ToyCircleDataset
from utils.visualization import plot_trajectories
from policies.flow_matcher import FlowMatcher
from models.dit import DiTPolicy
from utils.seed import set_seed
from utils.visualization import plot_lasa_trajectories

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
def main():

    set_seed(42)
    config = load_config("configs/toy_2d.yaml")
    device = torch.device(config["training"]["device"])
    chunk_size = config["dataset"]["chunk_size"]
    action_dim = config["dataset"]["action_dim"]
    state_dim = config["dataset"]["state_dim"]
    batch_size = config["training"]["batch_size"]
    lr = float(config["training"]["lr"])
    epochs = config["training"]["epochs"]
    hidden_dim = config["model"]["hidden_dim"]
    num_layers = config["model"]["num_layers"]

    print(f"Using device: {device}")

    # Data
    dataset = ToyCircleDataset(num_samples=2000, chunk_size = chunk_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model & Policy
    model = DiTPolicy(action_dim, state_dim, chunk_size, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    policy = FlowMatcher(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for state, action_chunk in dataloader:
            state = state.to(device)
            action_chunk = action_chunk.to(device)

            optimizer.zero_grad()
            loss = policy.compute_loss(action_chunk, state)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "weights/dit_toy.pt")
    print("Training complete. Model saved to weights/dit_toy.pt")

if __name__ == "__main__":
    main()