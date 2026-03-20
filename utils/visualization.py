import pyLasaDataset as lasa
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

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
    plt.axis('equal') # Keeps the circles looking circular

    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")


def plot_lasa_trajectories(pattern_names, train_indices, test_indices, generated_paths=None, save_path="lasa_viz.png"):
    pattern_data = getattr(lasa.DataSet, pattern_names)
    plt.figure(figsize=(8, 6))

    # Plot training demos
    for idx in train_indices:
        pos = pattern_data.demos[idx].pos
        plt.plot(pos[0, :], pos[1, :], 'b-', alpha=0.4, label='Train' if idx == train_indices[0] else "")

    # Plot testing demos
    for idx in test_indices:
        pos = pattern_data.demos[idx].pos
        plt.plot(pos[0, :], pos[1, :], 'g--', alpha=0.4, label='Test' if idx == test_indices[0] else "")

    # Plot generated paths if provided
    if generated_paths is not None:
        for i, path in enumerate(generated_paths):
            plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Generated' if i == 0 else "")

    plt.title(f"LASA Dataset: {pattern_names}")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()