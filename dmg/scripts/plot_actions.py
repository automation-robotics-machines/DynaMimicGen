import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import dmg.demos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help=("Path to your demonstration directory that contains the demo.hdf5 file"))
    parser.add_argument("--mimicgen", action="store_true")
    parser.add_argument("--play-dmp", action="store_true")
    parser.add_argument("--delta-actions", action="store_true")
    args = parser.parse_args()

    # Resolve HDF5 path
    if args.mimicgen:
        demo_path = os.path.join(dmg.demos.hdf5_root, "mimicgen", args.directory)
        if args.play_dmp:
            hdf5_path = os.path.join(demo_path, "dmp", "demo.hdf5")
        else:
            hdf5_path = os.path.join(demo_path, "demo.hdf5")
    else:
        demo_path = os.path.join(dmg.demos.hdf5_root, args.directory)
        if args.play_dmp:
            hdf5_path = os.path.join(demo_path, "dmp/demo.hdf5")
        else:
            hdf5_path = os.path.join(demo_path, "demo.hdf5")

    group_key = "data"
    action_key = "actions_abs"
    if (args.delta_actions and args.mimicgen) or (not args.mimicgen and not args.play_dmp):
        action_key = "actions"

    # Load all actions
    all_actions = []

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(
            list(f[group_key].keys()), key=lambda x: int(x.split("_")[1])
        )
        demo_keys = demo_keys[:10]
        for demo in demo_keys:
            actions = f[f"{group_key}/{demo}"][action_key][:]
            prev_action = actions[0, :].copy()
            for i in range(actions.shape[0]):
                for j in range(3, 6):
                    if abs(actions[i, j] - prev_action[j]) > 0.5:
                        actions[i, j] = -actions[i, j].copy()
                prev_action = actions[i, :].copy()
            all_actions.append(actions)

    # Stack into one big array: shape (total_timesteps, 7)
    all_actions = np.vstack(all_actions)
    print("✅ Loaded actions shape:", all_actions.shape)


    # Action dimensions
    action_labels = ["x", "y", "z", "rx", "ry", "rz", "gripper"]

    # Plotting
    fig, axs = plt.subplots(7, 2, figsize=(12, 18))
    fig.suptitle("Action Distribution per DOF", fontsize=16)

    for i in range(7):
        # Scatter plot (index vs. value)
        axs[i, 0].scatter(
            range(all_actions.shape[0]), all_actions[:, i], s=1, alpha=0.4
        )
        axs[i, 0].set_ylabel(action_labels[i])
        axs[i, 0].set_title(f"{action_labels[i]} - Scatter")

        # Histogram (distribution)
        axs[i, 1].hist(all_actions[:, i], bins=100, alpha=0.7, color="orange")
        axs[i, 1].set_title(f"{action_labels[i]} - Histogram")
        # axs[i, 1].set_ylim((0.0, 50.0))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # plt.show()

     # Normalize (Z-score normalization)
    means = np.mean(all_actions, axis=0)
    stds = np.std(all_actions, axis=0)
    normalized_actions = (all_actions - means) / stds

    print("ℹ️  Means before normalization:", means)
    print("ℹ️  Stddevs before normalization:", stds)

    # Action dimensions
    action_labels = ["x", "y", "z", "rx", "ry", "rz", "gripper"]

    # Plotting
    fig, axs = plt.subplots(7, 1, figsize=(10, 18))
    fig.suptitle("Normalized Action Distribution per DOF", fontsize=16)

    for i in range(7):
        axs[i].hist(normalized_actions[:, i], bins=100, alpha=0.7, color="steelblue", density=True)
        axs[i].set_title(f"{action_labels[i]} (Normalized)")
        axs[i].axvline(0, color="red", linestyle="--", linewidth=1)
        axs[i].set_ylabel("Density")

    plt.xlabel("Normalized Value")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
