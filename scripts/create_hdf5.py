import h5py
import numpy as np
import pyLasaDataset as lasa
import os
import json
import argparse
import matplotlib.pyplot as plt
import io
from PIL import Image

def render_trajectory_to_image(pos, size=(84, 84)):
    """Renders the 2D trajectory to an RGB numpy array."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(pos[:, 0], pos[:, 1], 'k-', linewidth=4)
    ax.axis('off')
    ax.set_aspect('equal')

    # Save to a buffer instead of disk
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # Read back as an image, resize, and convert to numpy
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img = img.resize(size, Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)

def create_robomimic_hdf5(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pattern_names = lasa.dataset.NAMES_

    with h5py.File(output_path, 'w') as h5f:
        data_grp = h5f.create_group("data")
        mask_grp = h5f.create_group("mask")

        total_samples = 0
        global_demo_idx = 0

        for pattern_name in pattern_names:
            pattern_data = getattr(lasa.DataSet, pattern_name)
            pattern_demo_names = []

            print(f"Processing {pattern_name}...")
            for demo in pattern_data.demos:
                demo_name = f"demo_{global_demo_idx}"
                pattern_demo_names.append(demo_name)
                demo_grp = data_grp.create_group(demo_name)

                pos = demo.pos.T.astype(np.float32)
                vel = demo.vel.T.astype(np.float32)
                acc = demo.acc.T.astype(np.float32)
                num_samples = len(pos)

                demo_grp.attrs["num_samples"] = num_samples
                total_samples += num_samples

                demo_grp.create_dataset("states", data=np.concatenate([pos, vel], axis=-1))
                demo_grp.create_dataset("actions", data=vel)
                demo_grp.create_dataset("rewards", data=np.zeros(num_samples, dtype=np.float32))

                dones = np.zeros(num_samples, dtype=np.float32)
                dones[-1] = 1.0
                demo_grp.create_dataset("dones", data=dones)

                obs_grp = demo_grp.create_group("obs")
                obs_grp.create_dataset("pos", data=pos)
                obs_grp.create_dataset("vel", data=vel)
                obs_grp.create_dataset("acc", data=acc)

                # Generate a single 84x84 RGB image of the shape
                img_array = render_trajectory_to_image(pos, size=(84, 84))

                # Tile it N times to match the timesteps: shape (N, 84, 84, 3)
                video_array = np.tile(img_array[None, ...], (num_samples, 1, 1, 1))

                # Save with gzip compression to prevent the file size from exploding
                obs_grp.create_dataset("image", data=video_array, compression="gzip")

                global_demo_idx += 1

            mask_grp.create_dataset(pattern_name, data=np.array(pattern_demo_names, dtype="S"))

        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps({
            "env_name": "LASA_MultiTask_Vision",
            "env_type": "custom",
            "env_kwargs": {}
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/lasa_vision.hdf5")
    args = parser.parse_args()
    create_robomimic_hdf5(args.output)
    print(f"Vision HDF5 created at: {args.output}")