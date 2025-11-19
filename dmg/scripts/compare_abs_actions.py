import argparse
import json
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import dmg.demos
from dmg.utils import from_env_args_to_env_info, unwrap_orientations, convert_delta_to_absolute, pose_to_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare HDF5 absolute actions vs delta→absolute reconstruction.")
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Path to demonstration directory (relative to dmg.demos.hdf5_root)",
    )
    parser.add_argument("--demo", type=str, default="demo_0", help="Demo name (e.g. demo_0)")
    args = parser.parse_args()

    # === Load demonstration ===
    demo_path = os.path.join(dmg.demos.hdf5_root, args.directory)
    hdf5_path = os.path.join(demo_path, "demo.hdf5")

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]

        # Load environment info (if available)
        try:
            env_info = json.loads(data_group.attrs["env_info"])
        except (KeyError, json.JSONDecodeError):
            env_info = from_env_args_to_env_info(json.loads(data_group.attrs["env_args"]))

        # Detect delta vs absolute control mode
        control_is_delta = env_info["controller_configs"].get("control_delta", False)
        print(f"Control mode: {'delta' if control_is_delta else 'absolute'}")

        # Load both actions if available
        abs_actions = np.array(f[f"data/{args.demo}/actions_abs"][()])
        delta_actions = np.array(f[f"data/{args.demo}/actions"][()])

    # === Convert delta → absolute ===
    # T_init = pose_to_transform(abs_actions[0, :6])
    abs_reconstructed = convert_delta_to_absolute(delta_actions, init_pose=abs_actions[0, :])

    # === Unwrap orientations to avoid 2π discontinuities ===
    abs_actions[:, 3:6] = unwrap_orientations(abs_actions[:, 3:6])
    abs_reconstructed[:, 3:6] = unwrap_orientations(abs_reconstructed[:, 3:6])

    # === Align lengths ===
    n = min(len(abs_actions), len(abs_reconstructed))
    abs_actions = abs_actions[:n]
    abs_reconstructed = abs_reconstructed[:n]

    # === Compute errors ===
    pos_diff = abs_actions[:, :3] - abs_reconstructed[:, :3]
    rot_diff = abs_actions[:, 3:6] - abs_reconstructed[:, 3:6]
    grip_diff = abs_actions[:, 6] - abs_reconstructed[:, 6]

    print(f"Mean position error: {np.mean(np.linalg.norm(pos_diff, axis=1)):.6f} m")
    print(f"Mean orientation error: {np.mean(np.linalg.norm(rot_diff, axis=1)):.6f} rad")
    print(f"Mean gripper error: {np.mean(np.abs(grip_diff)):.6f}")

    # === Plot comparisons ===
    labels_pos = ["X", "Y", "Z"]
    labels_rot = ["Rx", "Ry", "Rz"]

    # --- Positions ---
    fig_p, axs_p = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axs_p):
        ax.plot(abs_actions[:, i], "--", color="tab:red", label="Recorded abs")
        ax.plot(abs_reconstructed[:, i], "-", color="tab:blue", label="From delta→abs")
        ax.set_ylabel(f"{labels_pos[i]} [m]")
        ax.legend()
        ax.grid(True)
    axs_p[-1].set_xlabel("Timestep")
    fig_p.suptitle(f"Position Comparison — {args.demo}")
    plt.tight_layout()

    # --- Orientations ---
    fig_o, axs_o = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axs_o):
        ax.plot(abs_actions[:, 3 + i], "--", color="tab:red", label="Recorded abs")
        ax.plot(abs_reconstructed[:, 3 + i], "-", color="tab:blue", label="From delta→abs")
        ax.set_ylabel(f"{labels_rot[i]} [rad]")
        ax.legend()
        ax.grid(True)
    axs_o[-1].set_xlabel("Timestep")
    fig_o.suptitle(f"Orientation Comparison — {args.demo}")
    plt.tight_layout()

    # --- Gripper ---
    plt.figure(figsize=(10, 4))
    plt.plot(abs_actions[:, 6], "--", color="tab:red", label="Recorded abs")
    plt.plot(abs_reconstructed[:, 6], "-", color="tab:blue", label="From delta→abs")
    plt.title(f"Gripper Comparison — {args.demo}")
    plt.ylabel("Gripper")
    plt.xlabel("Timestep")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
