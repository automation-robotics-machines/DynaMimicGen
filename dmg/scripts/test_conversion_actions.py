import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import argparse

import dmg.demos
from dmg import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="test_delta", help="Directory of the demo to analyze")
    parser.add_argument("--demo", type=str, default="demo_0", help="Name of the demo to analyze")
    args = parser.parse_args()

    # === Load demonstration file ===
    demo_path = os.path.join(dmg.demos.hdf5_root, args.directory)
    hdf5_path = os.path.join(demo_path, "image.hdf5")

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        abs_actions = np.array(f[f"data/{args.demo}/actions_abs"][()])
        delta_actions = np.array(f[f"data/{args.demo}/actions"][()])

    print(f"Loaded {len(abs_actions)} absolute and {len(delta_actions)} delta actions from {args.demo}")

    # === 1️⃣ Convert ABS → DELTA → ABS (step-by-step) ===
    delta_from_abs = []
    for t in range(len(abs_actions)):
        prev = abs_actions[t - 1] if t > 0 else abs_actions[0]
        delta_from_abs.append(utils.convert_abs_action_to_delta_action(abs_actions[t], prev))
    delta_from_abs = np.stack(delta_from_abs)

    abs_reconstructed_from_abs = []
    prev_abs = abs_actions[0]
    for t in range(len(delta_from_abs)):
        abs_reconstructed_from_abs.append(utils.convert_delta_action_to_absolute_action(delta_from_abs[t], prev_abs))
        prev_abs = abs_reconstructed_from_abs[-1]
    abs_reconstructed_from_abs = np.stack(abs_reconstructed_from_abs)

    # === 2️⃣ Convert DELTA → ABS → DELTA (step-by-step) ===
    abs_from_delta = []
    prev_abs = np.zeros((7,))
    for t in range(len(delta_actions)):
        abs_from_delta.append(utils.convert_delta_action_to_absolute_action(delta_actions[t], prev_abs))
        prev_abs = abs_from_delta[-1]
    abs_from_delta = np.stack(abs_from_delta)

    delta_reconstructed_from_delta = []
    for t in range(len(abs_from_delta)):
        prev = abs_from_delta[t - 1] if t > 0 else abs_from_delta[0]
        delta_reconstructed_from_delta.append(utils.convert_abs_action_to_delta_action(abs_from_delta[t], prev))
    delta_reconstructed_from_delta = np.stack(delta_reconstructed_from_delta)

    # === Unwrap orientations to avoid jumps ===
    abs_actions[:, 3:6] = utils.unwrap_orientations(abs_actions[:, 3:6])
    abs_reconstructed_from_abs[:, 3:6] = utils.unwrap_orientations(abs_reconstructed_from_abs[:, 3:6])
    abs_from_delta[:, 3:6] = utils.unwrap_orientations(abs_from_delta[:, 3:6])

    # === Compute errors (A → D → A) ===
    n_abs = min(len(abs_actions), len(abs_reconstructed_from_abs))
    abs_actions = abs_actions[:n_abs]
    abs_reconstructed_from_abs = abs_reconstructed_from_abs[:n_abs]

    pos_diff_abs = np.linalg.norm(abs_actions[:, :3] - abs_reconstructed_from_abs[:, :3], axis=1)
    rot_diff_abs = np.linalg.norm(abs_actions[:, 3:6] - abs_reconstructed_from_abs[:, 3:6], axis=1)
    grip_diff_abs = np.abs(abs_actions[:, 6] - abs_reconstructed_from_abs[:, 6])

    print("\n=== Reversibility: ABS → DELTA → ABS ===")
    print(f"Mean position error:    {np.mean(pos_diff_abs):.6e} m")
    print(f"Mean orientation error: {np.mean(rot_diff_abs):.6e} rad")
    print(f"Mean gripper error:     {np.mean(grip_diff_abs):.6e}")

    # === Compute errors (D → A → D) ===
    n_delta = min(len(delta_actions), len(delta_reconstructed_from_delta))
    delta_actions = delta_actions[:n_delta]
    delta_reconstructed_from_delta = delta_reconstructed_from_delta[:n_delta]

    pos_diff_delta = np.linalg.norm(delta_actions[:, :3] - delta_reconstructed_from_delta[:, :3], axis=1)
    rot_diff_delta = np.linalg.norm(delta_actions[:, 3:6] - delta_reconstructed_from_delta[:, 3:6], axis=1)
    grip_diff_delta = np.abs(delta_actions[:, 6] - delta_reconstructed_from_delta[:, 6])

    print("\n=== Reversibility: DELTA → ABS → DELTA ===")
    print(f"Mean position error:    {np.mean(pos_diff_delta):.6e} m")
    print(f"Mean orientation error: {np.mean(rot_diff_delta):.6e} rad")
    print(f"Mean gripper error:     {np.mean(grip_diff_delta):.6e}")

    # === Plot ABS → DELTA → ABS comparison ===
    fig_p, axs_p = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    pos_labels = ["X", "Y", "Z"]
    for i, ax in enumerate(axs_p):
        ax.plot(abs_actions[:, i], "--", color="tab:red", label="Original abs")
        ax.plot(abs_reconstructed_from_abs[:, i], "-", color="tab:blue", label="Reconstructed abs")
        ax.set_ylabel(f"{pos_labels[i]} [m]")
        ax.legend()
        ax.grid(True)
    axs_p[-1].set_xlabel("Timestep")
    fig_p.suptitle(f"ABS → DELTA → ABS Position Components ({args.demo})")
    plt.tight_layout()

    fig_o, axs_o = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    rot_labels = ["Rx", "Ry", "Rz"]
    for i, ax in enumerate(axs_o):
        ax.plot(abs_actions[:, 3 + i], "--", color="tab:red", label="Original abs")
        ax.plot(abs_reconstructed_from_abs[:, 3 + i], "-", color="tab:blue", label="Reconstructed abs")
        ax.set_ylabel(f"{rot_labels[i]} [rad]")
        ax.legend()
        ax.grid(True)
    axs_o[-1].set_xlabel("Timestep")
    fig_o.suptitle(f"ABS → DELTA → ABS Orientation Components ({args.demo})")
    plt.tight_layout()

    # === Plot DELTA → ABS → DELTA comparison ===
    fig_d, axs_d = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axs_d):
        ax.plot(delta_actions[:, i], "--", color="tab:green", label="Original delta")
        ax.plot(delta_reconstructed_from_delta[:, i], "-", color="tab:purple", label="Reconstructed delta")
        ax.set_ylabel(f"Δ{pos_labels[i]} [m]")
        ax.legend()
        ax.grid(True)
    axs_d[-1].set_xlabel("Timestep")
    fig_d.suptitle(f"DELTA → ABS → DELTA Position Components ({args.demo})")
    plt.tight_layout()

    fig_c, axs_c = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axs_c):
        ax.plot(delta_actions[:, 3 + i], "--", color="tab:red", label="Original delta")
        ax.plot(delta_reconstructed_from_delta[:, 3 + i], "-", color="tab:blue", label="Reconstructed delta")
        ax.set_ylabel(f"{rot_labels[i]} [rad]")
        ax.legend()
        ax.grid(True)
    axs_c[-1].set_xlabel("Timestep")
    fig_c.suptitle(f"DELTA → ABS → DELTA Orientation Components ({args.demo})")
    plt.tight_layout()

    # === Error over time (A → D → A) ===
    fig_e, axs_e = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axs_e[0].plot(pos_diff_abs, color="tab:green")
    axs_e[0].set_ylabel("‖Δpos‖ [m]")
    axs_e[1].plot(rot_diff_abs, color="tab:orange")
    axs_e[1].set_ylabel("‖Δrot‖ [rad]")
    axs_e[2].plot(grip_diff_abs, color="tab:purple")
    axs_e[2].set_ylabel("‖Δgrip‖")
    axs_e[2].set_xlabel("Timestep")
    fig_e.suptitle(f"Error Over Time — ABS → DELTA → ABS ({args.demo})")
    plt.tight_layout()

    # === Error over time (D → A → D) ===
    fig_ed, axs_ed = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axs_ed[0].plot(pos_diff_delta, color="tab:green")
    axs_ed[0].set_ylabel("‖Δpos‖ [m]")
    axs_ed[1].plot(rot_diff_delta, color="tab:orange")
    axs_ed[1].set_ylabel("‖Δrot‖ [rad]")
    axs_ed[2].plot(grip_diff_delta, color="tab:purple")
    axs_ed[2].set_ylabel("‖Δgrip‖")
    axs_ed[2].set_xlabel("Timestep")
    fig_ed.suptitle(f"Error Over Time — DELTA → ABS → DELTA ({args.demo})")
    plt.tight_layout()

    plt.show()
