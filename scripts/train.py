import sys
import os
import yaml
import torch
import wandb
import argparse
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dit import DiTPolicy
from policies.flow_matcher import FlowMatcher
from utils.config import load_config, build_run_name
from utils.seed import set_seed
from utils.visualization import plot_lasa_trajectories
from utils.normalizer import build_normalizer
from data.datasets import build_datasets


def train_one_epoch(policy, loader, normalizer, optimizer, device) -> float:
    policy.train()
    total_loss = 0.0
    for state, action_chunk in loader:
        state = normalizer.normalize('state', state.to(device))
        action_chunk = normalizer.normalize('action', action_chunk.to(device))
        optimizer.zero_grad()
        loss = policy.compute_loss(action_chunk, state)
        loss.backward()
        optimizer.step()
        total_loss = + loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(policy, loader, normalizer, device) -> float:
    policy.eval()
    total_loss = 0.0
    for state, action_chunk in loader:
        state = normalizer.normalize('state', state.to(device))
        action_chunk = normalizer.normalize('action', action_chunk.to(device))
        loss = policy.compute_loss(action_chunk, state)
        total_loss += loss.item()
    return total_loss / len(loader)


def save_checkpoint(model, normalizer, epoch, val_loss, path) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "normalizer": normalizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss
        },
        path,
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    set_seed(42)
    config = load_config(args.config)

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    dataset_name = config["dataset"].get("name", "lasa").lower()
    run_name = build_run_name(config)

    wandb.init(project="lasa_flow_matching", name=run_name, config=config)

    # ── Data ────────────────────────────────────────────────────────────────
    train_dataset, val_dataset = build_datasets(config)
    normalizer = build_normalizer(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = DiTPolicy(
        action_dim=config["dataset"]["action_dim"],
        state_dim=config["dataset"]["state_dim"],
        chunk_size=config["dataset"]["chunk_size"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"].get("num_heads", 4),
    ).to(device)

    policy = FlowMatcher(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

    epochs = config["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ───────────────────────────────────────────────────────
    os.makedirs("weights", exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = f"weights/{run_name}_best.pt"

    for epoch in range(epochs):
        train_loss = train_one_epoch(policy, train_loader, normalizer, optimizer, device)
        val_loss = evaluate(policy, val_loader, normalizer, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, normalizer, epoch, best_val_loss, best_ckpt_path)
            print(f"  -> Best model saved (val loss: {best_val_loss:.4f})")

    # ── Save final weights ──────────────────────────────────────────────────
    final_path = f"weights/{config.get('project_name', dataset_name)}.pt"
    torch.save({"model": model.state_dict(), "normalizer": normalizer.state_dict()}, final_path)
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    # ── Visualize & log ─────────────────────────────────────────────────────
    if dataset_name == "lasa":
        os.makedirs("results", exist_ok=True)
        img_path = f"results/{run_name}_base_data.png"
        plot_lasa_trajectories(
            config["dataset"]["pattern_name"],
            config["dataset"]["train_indices"],
            config["dataset"]["test_indices"],
            save_path=img_path,
        )
        wandb.log({"ground_truth_visualization": wandb.Image(img_path)})

    wandb.finish()


if __name__ == "__main__":
    main()