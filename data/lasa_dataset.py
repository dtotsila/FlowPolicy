import torch
from torch.utils.data import Dataset
import pyLasaDataset as lasa

class LasaDataset(Dataset):
    def __init__(self, pattern_name="Angle", chunk_size=16, demo_indices=None):
        self.chunk_size = chunk_size
        self.samples = []

        # Default to all 7 if none specified
        if demo_indices is None:
            demo_indices = list(range(7))

        pattern_data = getattr(lasa.DataSet, pattern_name)

        # Only load the specific demonstrations requested
        for idx in demo_indices:
            demo = pattern_data.demos[idx]
            pos = demo.pos.T

            for i in range(len(pos) - chunk_size):
                state = pos[i]
                chunk = pos[i : i + chunk_size]
                self.samples.append((state, chunk))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, chunk = self.samples[idx]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(chunk, dtype=torch.float32)