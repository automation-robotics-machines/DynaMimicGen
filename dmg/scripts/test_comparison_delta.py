import numpy as np
import os
import h5py
import argparse
import matplotlib.pyplot as plt

import dmg.demos
from dmg.utils import (
    convert_delta_to_absolute,
    get_initial_robot_matrix,
    unwrap_orientations,
    load_environment,
)
from robosuite.utils.transform_utils import axisangle2quat, quat2mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help="Path to the demonstration directory.")
    parser.add_argument("--num-dmp", type=int, default=200)
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--robot", type=str, default="default")
    parser.add_argument("--distr", type=str, default="D0")
    parser.add_argument("--camera-res", type=int, default=84)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dynamic-env", action="store_true")
    parser.add_argument("--perc-horizon", type=float, default=0.5)
    parser.add_argument("--delta-actions", action="store_true", help="Use delta actions for DMP training and rollout.")
    parser.add_argument("--demo", type=str, default="demo_0", help="Which demo to use.")
    args = parser.parse_args()

    # === Load HDF5 file ===
    demo_path = os.path.join(dmg.demos.hdf5_root, args.directory)
    hdf5_path = os.path.join(demo_path, "image.hdf5")
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    f = h5py.File(hdf5_path, "r")
    env, env_name, env_args, env_info = load_environment(f, args)
    env.reset()
    states = np.array(f[f"data/{args.demo}/states"][()])
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    # === Load action data ===
    delta_actions = np.array(f[f"data/{args.demo}/actions"][()])
    rel_pos = np.array(f[f"data/{args.demo}/action_dict/rel_pos"][()])
    rel_orient = np.array(f[f"data/{args.demo}/action_dict/rel_rot_axis_angle"][()])
    rel_gripper = np.array(f[f"data/{args.demo}/action_dict/gripper"][()])
    rel_actions = np.concatenate([rel_pos, rel_orient, rel_gripper], axis=1)

    abs_actions = np.array(f[f"data/{args.demo}/actions_abs"][()])
    abs_actions[:, 3:6] = unwrap_orientations(abs_actions[:, 3:6])
    f.close()

    # === Compare delta_actions vs rel_actions ===
    print(f"Comparing delta_actions and rel_actions ({args.demo})")

    min_len = min(len(delta_actions), len(rel_actions))
    delta_actions = delta_actions[:min_len]
    rel_actions = rel_actions[:min_len]

    delta_diff = np.abs(delta_actions - rel_actions)
    mean_diff = np.mean(delta_diff, axis=0)
    print(f"Mean per-dimension difference (delta vs rel): {mean_diff}")

    # === Plot comparison delta vs rel ===
    fig_d, axs_d = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["X", "Y", "Z"]
    for i, ax in enumerate(axs_d):
        ax.plot(delta_actions[:, i], "--", color="tab:red", label="delta_actions")
        ax.plot(rel_actions[:, i], "-", color="tab:blue", label="rel_actions")
        ax.set_ylabel(f"{labels[i]} [m]")
        ax.legend()
        ax.grid(True)
    axs_d[-1].set_xlabel("Timestep")
    fig_d.suptitle(f"Delta vs Rel Position Comparison ({args.demo})")
    plt.tight_layout()

    # === Plot orientation (axis-angle) delta vs rel ===
    fig_do, axs_do = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    rot_labels = ["Rx", "Ry", "Rz"]
    for i, ax in enumerate(axs_do):
        ax.plot(delta_actions[:, 3 + i], "--", color="tab:red", label="delta_actions")
        ax.plot(rel_actions[:, 3 + i], "-", color="tab:blue", label="rel_actions")
        ax.set_ylabel(f"{rot_labels[i]} [rad]")
        ax.legend()
        ax.grid(True)
    axs_do[-1].set_xlabel("Timestep")
    fig_do.suptitle(f"Delta vs Rel Orientation Comparison ({args.demo})")
    plt.tight_layout()

    # === Convert delta_actions → absolute trajectory ===
    T_init_robot = get_initial_robot_matrix(env)
    abs_trajectory = convert_delta_to_absolute(delta_actions, T_init_robot)
    abs_trajectory[:, 3:6] = unwrap_orientations(abs_trajectory[:, 3:6])

    # === Compare absolute trajectories ===
    print(f"Comparing reconstructed abs_trajectory and HDF5 abs_actions")

    min_len = min(len(abs_trajectory), len(abs_actions))
    abs_trajectory = abs_trajectory[:min_len]
    abs_actions = abs_actions[:min_len]

    pos_diff = np.linalg.norm(abs_trajectory[:, :3] - abs_actions[:, :3], axis=1)
    rot_diff = np.linalg.norm(abs_trajectory[:, 3:6] - abs_actions[:, 3:6], axis=1)

    print(f"Mean position error: {np.mean(pos_diff):.5f} m")
    print(f"Mean orientation error: {np.mean(rot_diff):.5f} rad")

    # === Plot absolute position comparison ===
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    pos_labels = ["X", "Y", "Z"]
    for i, ax in enumerate(axs1):
        ax.plot(abs_actions[:, i], "--", color="tab:red", label="HDF5 abs_actions")
        ax.plot(abs_trajectory[:, i], "-", color="tab:blue", label="Reconstructed from delta")
        ax.set_ylabel(f"{pos_labels[i]} [m]")
        ax.grid(True)
        ax.legend()
    axs1[-1].set_xlabel("Timestep")
    fig1.suptitle(f"Absolute Position Comparison ({args.demo})")
    plt.tight_layout()

    # === Plot absolute orientation comparison ===
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    rot_labels = ["Rx", "Ry", "Rz"]
    for i, ax in enumerate(axs2):
        ax.plot(abs_actions[:, 3 + i], "--", color="tab:red", label="HDF5 abs_actions")
        ax.plot(abs_trajectory[:, 3 + i], "-", color="tab:blue", label="Reconstructed from delta")
        ax.set_ylabel(f"{rot_labels[i]} [rad]")
        ax.grid(True)
        ax.legend()
    axs2[-1].set_xlabel("Timestep")
    fig2.suptitle(f"Absolute Orientation Comparison ({args.demo})")
    plt.tight_layout()

    # === Plot absolute reconstruction error ===
    fig3, axs3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs3[0].plot(pos_diff, color="tab:green")
    axs3[0].set_ylabel("‖Δpos‖ [m]")
    axs3[0].grid(True)
    axs3[1].plot(rot_diff, color="tab:orange")
    axs3[1].set_ylabel("‖Δrot‖ [rad]")
    axs3[1].set_xlabel("Timestep")
    axs3[1].grid(True)
    fig3.suptitle(f"Reconstruction Error Over Time ({args.demo})")
    plt.tight_layout()

    plt.show()
