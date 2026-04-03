import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

class LasaHDF5Dataset(Dataset):
    def __init__(self, hdf5_path, pattern_names, chunk_size=16, demo_indices=None,
                 use_velocity=False, use_acceleration=False, use_vision=False):

        self.chunk_size = chunk_size
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.use_vision = use_vision
        self.samples = []

        if demo_indices is None:
            demo_indices = list(range(7)) # LASA always has 7 demos per shape

        if isinstance(pattern_names, str):
            pattern_names = [pattern_names]

        # Read directly from our new HDF5 structure
        with h5py.File(hdf5_path, 'r') as f:
            for class_id, pattern in enumerate(pattern_names):

                # Fetch the list of demonstrations for this specific shape
                demo_names = f["mask"][pattern][()]

                for idx in demo_indices:
                    # Robomimic mask stores strings as bytes, decode to match group names
                    demo_name = demo_names[idx].decode('utf-8')
                    demo_grp = f["data"][demo_name]

                    pos = demo_grp["obs"]["pos"][:]
                    vel = demo_grp["obs"]["vel"][:]
                    acc = demo_grp["obs"]["acc"][:]

                    if self.use_vision:
                        images = demo_grp["obs"]["image"][:]
                        # Convert to PyTorch (N, C, H, W) and normalize pixels to [0, 1]
                        images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32) / 255.0

                    # Standard chunking sliding window
                    for i in range(len(pos) - chunk_size - 1):
                        state_components = [pos[i]]
                        chunk_components = [pos[i+1 : i+1+chunk_size]]

                        if self.use_velocity:
                            state_components.append(vel[i])
                            chunk_components.append(vel[i+1 : i+1+chunk_size])
                        if self.use_acceleration:
                            state_components.append(acc[i])
                            chunk_components.append(acc[i+1 : i+1+chunk_size])

                        state = np.concatenate(state_components)
                        chunk = np.concatenate(chunk_components, axis=1)

                        if self.use_vision:
                            image = images[i]
                            self.samples.append((state, chunk, image, class_id))
                        else:
                            self.samples.append((state, chunk, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_vision:
            state, chunk, image, label = self.samples[idx]
        else:
            state, chunk, label = self.samples[idx]

        state_tensor = torch.tensor(state, dtype=torch.float32)
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Retain your relative target calculation
        delta_chunk = chunk_tensor - state_tensor

        if self.use_vision:
            image_tensor = torch.tensor(image, dtype=torch.float32)
            return state_tensor, delta_chunk, image_tensor, label_tensor
        else:
            return state_tensor, delta_chunk, label_tensor


def build_datasets(config: dict):
    dataset_name = config["dataset"].get("name", "lasa").lower()
    use_vision = config["dataset"].get("use_vision", False)

    # Ensure you set "hdf5_path" in your config, fallback to default names
    default_path = "data/lasa_vision.hdf5" if use_vision else "data/lasa_robomimic.hdf5"
    hdf5_path = config["dataset"].get("hdf5_path", default_path)

    if "lasa" in dataset_name:
        kwargs = dict(
            hdf5_path=hdf5_path,
            pattern_names=config["dataset"]["pattern_names"],
            chunk_size=config["dataset"]["chunk_size"],
            use_velocity=config["dataset"].get("use_velocity", False),
            use_acceleration=config["dataset"].get("use_acceleration", False),
            use_vision=use_vision
        )
        train_dataset = LasaHDF5Dataset(demo_indices=config["dataset"]["train_indices"], **kwargs)
        val_dataset = LasaHDF5Dataset(demo_indices=config["dataset"]["val_indices"], **kwargs)

        return train_dataset, val_dataset

    else:
        # Keep any other toy datasets here if you still use them
        raise NotImplementedError("Only LASA HDF5 is currently supported.")