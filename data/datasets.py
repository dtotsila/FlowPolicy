import torch
import numpy as np
from torch.utils.data import Dataset
import pyLasaDataset as lasa

class LasaDataset(Dataset):
    def __init__(self, pattern_name="Angle", chunk_size=16, demo_indices=None, include_velocity=False, include_acceleration=False):
        self.chunk_size = chunk_size
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration
        self.samples = []

        if demo_indices is None:
            demo_indices = list(range(7))

        pattern_data = getattr(lasa.DataSet, pattern_name)

        for idx in demo_indices:
            demo = pattern_data.demos[idx]
            pos = demo.pos.T
            vel = demo.vel.T
            acc = demo.acc.T

            for i in range(len(pos) - chunk_size - 1):
                # Build state dynamically based on flags
                state_components = [pos[i]]
                chunk_components = [pos[i+1:i+1+chunk_size]]
                if self.include_velocity:
                    state_components.append(vel[i])
                    chunk_components.append(vel[i+1:i+1+chunk_size])
                if self.include_acceleration:
                    state_components.append(acc[i])
                    chunk_components.append(acc[i+1:i+1+chunk_size])

                state = np.concatenate(state_components)
                chunk = np.concatenate(chunk_components, axis=1)


                self.samples.append((state, chunk, pos[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, chunk, current_pos = self.samples[idx]

        state_tensor = torch.tensor(state, dtype=torch.float32)
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32)


        # Delta is always computed against position only (for now)
        delta_chunk = chunk_tensor - state_tensor

        return state_tensor, delta_chunk

class ToyCircleDataset(Dataset):
    def __init__(self, num_samples=2000, chunk_size=16):
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            start_angle = torch.rand(1) * 2 * np.pi
            # Generate chunk_size + 1 to account for the initial state
            time_steps = torch.linspace(0, 2 * np.pi, self.chunk_size + 1) + start_angle
            chunk = torch.stack([torch.cos(time_steps), torch.sin(time_steps)], dim=1)
            data.append(chunk)

        data = torch.stack(data)
        jitter = torch.randn_like(data) * 0.05
        return data + jitter

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        full_chunk = self.data[idx]
        state = full_chunk[0]
        chunk = full_chunk[1:] # The actual future steps

        delta_chunk = chunk - state

        return state, delta_chunk

def build_datasets(config: dict):
    dataset_name = config["dataset"].get("name", "lasa").lower()

    if dataset_name == "lasa":
        kwargs = dict(
            pattern_name=config["dataset"]["pattern_name"],
            chunk_size=config["dataset"]["chunk_size"],
            include_velocity=config["dataset"].get("use_velocity", False),
            include_acceleration=config["dataset"].get("use_acceleration", False),
        )
        train_dataset = LasaDataset(demo_indices=config["dataset"]["train_indices"], **kwargs)
        val_dataset = LasaDataset(demo_indices=config["dataset"]["val_indices"], **kwargs)
    else:
        train_dataset = ToyCircleDataset(
            num_samples=config["dataset"].get("num_samples", 2000),
            chunk_size=config["dataset"]["chunk_size"],
        )
        val_dataset = train_dataset  # toy dataset has no separate split

    return train_dataset, val_dataset