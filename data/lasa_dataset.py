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
            # We need to shift the chunk forward
            for i in range(len(pos) - chunk_size - 1):
                state = pos[i]
                # shift chunk to predict the next chunk
                chunk = pos[i+1: i + 1 + chunk_size]
                self.samples.append((state, chunk))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, chunk = self.samples[idx]

        state_tensor = torch.tensor(state, dtype=torch.float32)
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32)

        delta_chunk = chunk_tensor - state_tensor

        return state_tensor, delta_chunk