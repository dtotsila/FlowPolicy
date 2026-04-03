import pyLasaDataset as lasa
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import io
from PIL import Image
import torch
import numpy as np


def plot_trajectories(action_chunks, num_samples=5, save_path="results/dataset_viz.png"):
    plt.figure(figsize=(6, 6))

    # Plot a few chunks to see the base circle and the jitter
    for i in range(min(num_samples, len(action_chunks))):
        chunk = action_chunks[i].numpy()
        plt.plot(chunk[:, 0], chunk[:, 1], marker='o', markersize=4, alpha=0.6)

    plt.title("Toy 2D Circular Trajectories (Action Chunks)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.axis('equal')  # Keeps the circles looking circular

    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")


def plot_lasa_trajectories(pattern_names, train_indices, test_indices, generated_paths=None, save_path="lasa_viz.png"):
    pattern_data = getattr(lasa.DataSet, pattern_names)
    plt.figure(figsize=(8, 6))

    # Plot training demos
    for idx in train_indices:
        pos = pattern_data.demos[idx].pos
        plt.plot(pos[0, :], pos[1, :], 'b-', alpha=0.4,
                 label='Train' if idx == train_indices[0] else "")

    # Plot testing demos
    for idx in test_indices:
        pos = pattern_data.demos[idx].pos
        plt.plot(pos[0, :], pos[1, :], 'g--', alpha=0.4,
                 label='Test' if idx == test_indices[0] else "")

    # Plot generated paths if provided
    if generated_paths is not None:
        for i, path in enumerate(generated_paths):
            plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2,
                     label='Generated' if i == 0 else "")

    plt.title(f"LASA Dataset: {pattern_names}")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def render_trajectory_to_image(pos, size=(84, 84)):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(pos[:, 0], pos[:, 1], 'k-', linewidth=4)
    ax.axis('off')
    ax.set_aspect('equal')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img = img.resize(size, Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)


def plot_streamlines(ax, policy, normalizer, pattern_data, device, chunk_size, num_inference_steps, class_id=None, image=None, grid_size=40):
    all_pos = np.concatenate([d.pos.T for d in pattern_data.demos])
    x_min, x_max = all_pos[:, 0].min() - 5, all_pos[:, 0].max() + 5
    y_min, y_max = all_pos[:, 1].min() - 5, all_pos[:, 1].max() + 5

    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    pts = np.c_[X.ravel(), Y.ravel()]

    U = np.zeros(len(pts))
    V = np.zeros(len(pts))
    batch_size = 256

    print(f"Generating streamlines for {ax.get_title()}...")
    with torch.no_grad():
        for i in range(0, len(pts), batch_size):
            p = torch.tensor(pts[i:i+batch_size],
                             dtype=torch.float32).to(device)
            norm_p = normalizer.normalize('state', p)
            current_batch_size = p.shape[0]

            cond = torch.full((current_batch_size,), class_id, dtype=torch.long,
                              device=device) if class_id is not None else None
            batched_image = image.expand(
                current_batch_size, -1, -1, -1) if image is not None else None

            norm_chunk = policy.sample(
                norm_p, chunk_size, sampling_steps=num_inference_steps, condition=cond, image=batched_image)
            chunk = normalizer.denormalize('action', norm_chunk).cpu().numpy()

            U[i:i+batch_size] = chunk[:, 0, 0]
            V[i:i+batch_size] = chunk[:, 0, 1]

    ax.streamplot(X, Y, U.reshape(grid_size, grid_size), V.reshape(grid_size, grid_size),
                  color='lightgray', density=1.5, linewidth=0.8, arrowsize=1.0, zorder=0)
