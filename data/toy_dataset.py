import torch
import math
from torch.utils.data import Dataset

class ToyCircleDataset(Dataset):
    def __init__(self, num_samples=2000, chunk_size=16):
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Start each circle at a random angle
            start_angle = torch.rand(1) * 2 * math.pi
            # Generate the next chunk_size steps from that angle
            time_steps = torch.linspace(0, 2 * math.pi, self.chunk_size) + start_angle
            chunk = torch.stack([torch.cos(time_steps), torch.sin(time_steps)], dim=1)
            data.append(chunk)

        data = torch.stack(data)
        jitter = torch.randn_like(data) * 0.05
        return data + jitter

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        action_chunk = self.data[idx]
        # State is now the starting (X, Y) coordinate of this chunk
        state = action_chunk[0]
        return state, action_chunk