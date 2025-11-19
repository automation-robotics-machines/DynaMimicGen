import os
import json
import argparse
import numpy as np
import h5py
import copy
import time
import warnings
import matplotlib.pyplot as plt

# robosuite transforms
from robosuite.utils.transform_utils import (
    quat2mat,
    mat2quat,
    axisangle2quat,
    quat2axisangle,
)

import dmg.demos
from dmg.dmp import JointDMP

# Try to import utilities from dmg.utils that the script expects.
# If any are missing, provide a helpful error or a fallback.
from dmg import utils

def train_single_dmp(traj, env_name):
    """Train a list of JointDMPs from subtask action segments."""
    dt = 0.002
    tau = traj.shape[0] * dt
    ts = np.arange(0, tau, dt)
    alpha = 25.0
    beta = alpha / 4
    cs_alpha = -np.log(0.0001)

    dmp = JointDMP(
        NDOF=traj.shape[1],
        n_bfs=100,
        alpha=alpha,
        beta=beta,
        cs_alpha=cs_alpha
    )
    dmp.train(traj.copy(), ts.copy(), tau)

    return dmp, ts, tau

def rollout_dmp_to_trajectory(dmp, ts, tau):
    """
    Rollout a DMP to generate a trajectory given time steps and tau.
    Assumes dmp has a rollout method that accepts ts and tau.
    """
    # Create a copy of the DMP to avoid modifying the original during rollout
    dmp_copy = copy.deepcopy(dmp)
    # Rollout the DMP
    p, dp, ddp = dmp_copy.rollout(ts=ts, tau=tau, FX=True)

    return p  # shape (T, dof)

def set_dmp_goal_from_transform(dmp, H_goal):
    """
    Set the DMP goal for a DMP object from a homogeneous transform H_goal.
    Assumes the DMP expects the pose as a vector (x,y,z,rx,ry,rz).
    The internal attribute name for goal may differ; we try `gp` (as in your code)
    and fall back to dmp.set_goal() if available.
    """
    goal_vec = utils.transform_to_pose(H_goal)
    # Many DMP implementations use dmp.gp for goal pose; adapt if needed.
    if hasattr(dmp, "gp"):
        dmp.gp = goal_vec.copy()
    elif hasattr(dmp, "set_goal"):
        dmp.set_goal(goal_vec.copy())
    else:
        # set attribute anyway
        dmp.gp = goal_vec.copy()
    return dmp

def main(args):
    demo_dir = os.path.join(dmg.demos.hdf5_root, args.directory)
    hdf5_path = os.path.join(demo_dir, "image.hdf5")
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    # open file
    f = h5py.File(hdf5_path, "r")

    ep = args.demo  # demo key like "demo_0"
    if f"data/{ep}/actions" not in f:
        raise KeyError(f"Demo actions not found: data/{ep}/actions")

    # 2) Load delta_actions array from data/demo_0 group
    delta_actions = np.array(f[f"data/{ep}/actions"][()])  # shape (T, 7) expected: dx,dy,dz,dRx,dRy,dRz,grip
    init_robot_pos = np.array(f[f"data/{ep}/obs/robot0_eef_pos"][()])[0, :]  # initial robot pose for this demo
    init_robot_quat = np.array(f[f"data/{ep}/obs/robot0_eef_quat"][()])[0, :]  # initial robot orientation for this demo
    init_robot_rotmat = quat2mat(init_robot_quat)
    T_init_robot = np.eye(4)
    T_init_robot[:3, 3] = init_robot_pos
    T_init_robot[:3, :3] = init_robot_rotmat  

    # 3) Convert delta_actions to abs_actions
    # Load environment
    env, env_name, env_args = utils.load_environment(f, args)
    abs_actions = utils.convert_delta_to_absolute(delta_actions)
    abs_actions[:, 3:6] = utils.unwrap_orientations(abs_actions[:, 3:6])

    # 4) Load position and orientation of the object to be grasped
    # subtask_data extraction
    if f"data/{ep}" not in f:
        raise KeyError(f"Demo {ep} not found in file.")
    subtask_data = json.loads(f[f"data/{ep}"].attrs["subtask_data"])
    ts_split = subtask_data["ts_split"]
    num_dmps = len(subtask_data["subtask_names"])

    subtask_idx = subtask_data["subtask_names"].index("Pick")

    object_ = np.array(f[f"data/{ep}/obs/object"][()])

    # use util helper to select object pose for this subtask
    object_pos, object_quat = utils.get_object_pose_for_env(env_name, env_args, object_, subtask_data, subtask_idx)
    # Build H_object_world
    H_object_demo = utils.posmat2mat(object_pos, quat2mat(object_quat))
    # subtask absolute actions from previously reconstructed abs_actions
    start_idx = ts_split[subtask_idx]
    end_idx = ts_split[subtask_idx + 1]
    # abs_actions shape assumed (T, 7): [x,y,z, rx,ry,rz, gripper]
    pick_traj = abs_actions[start_idx:end_idx, :-1].copy()
    demo_delta = delta_actions[start_idx:end_idx, :].copy()
    

    # 5) Train DMPs: use provided train_dmps helper (expects list of segments and env_name)
    trained_dmp, ts, tau = train_single_dmp(pick_traj, env_name)

    # 6) Generate new object position randomly on the table
    if args.random_pose:
        # new_x, new_y, new_zrot = utils.generate_object_position(env_name)
        # # build new object transform (we need a quaternion for object orientation)
        # # simple approach: use z-axis rotation quaternion and zero roll/pitch for the object
        # # construct quaternion from axis-angle: rotation about z by new_zrot
        # qz = axisangle2quat(np.array([0.0, 0.0, new_zrot]))  # axis-angle -> quaternion
        # new_object_pos = np.array([new_x, new_y, 0.0])  # assume table z = 0.0; adjust if needed
        original_object_pos, original_object_quat = utils.get_object_pose_for_env(env_name, env_args, object_, subtask_data, 0)
        new_object_pos = original_object_pos.copy()
        # new_object_pos[0] += 0.1  # small random x offset
        qz = original_object_quat.copy()
    else:
        # keep original orientation
        original_object_pos, original_object_quat = utils.get_object_pose_for_env(env_name, env_args, object_, subtask_data, 0)
        new_object_pos = original_object_pos.copy()
        qz = original_object_quat.copy()
    H_object_new = utils.posmat2mat(new_object_pos, quat2mat(qz))

    # 7) For each DMP compute demo gripper->object relative transform and set new goal
    # We'll compute H_object_demo and H_gripper_demo for each subtask (last pose of segment)

    
    K = pick_traj.shape[0]
    # final gripper pose in demo
    demo_pos = pick_traj[-1, :3].copy()
    demo_rot_mat = quat2mat(axisangle2quat(pick_traj[-1, 3:6].copy()))
    H_gripper_demo = utils.posmat2mat(demo_pos, demo_rot_mat)

    # compute H_object_demo: use subtask object's pose for this subtask
    object_pos, object_quat = utils.get_object_pose_for_env(env_name, env_args, object_, subtask_data, subtask_idx)
    H_object_demo = utils.posmat2mat(object_pos, quat2mat(object_quat))

    # relative transform: H_object_gripper such that H_gripper_demo = H_object_demo * H_object_gripper
    H_object_gripper = utils.pose_in_A_to_pose_in_B(H_gripper_demo.copy(), utils.pose_inv(H_object_demo.copy()))

    # new gripper goal in world frame
    H_gripper_new = H_object_new @ H_object_gripper

    # set the DMP goal (assume dmp has gp or set_goal)
    set_dmp_goal_from_transform(trained_dmp, H_gripper_new)

    # get ts and tau for this dmp
    ts = ts
    tau = tau
    # Rollout DMP -> absolute delta positions for DOF
    p_traj = rollout_dmp_to_trajectory(trained_dmp, ts, tau)  # shape (K, DOF)
    # Append gripper action sequence from original demo (we reuse gripper commands).
    gripper_seq = delta_actions[start_idx:end_idx, -1] if "start_idx" in locals() else delta_actions[:K, -1]
    # If that fails fall back to using final gripper state
    if len(gripper_seq) != p_traj.shape[0]:
        gripper_seq = np.full((p_traj.shape[0],), fill_value=delta_actions[-1, -1])

    # Form full (DOF + gripper) actions - these are absolute poses with gripper
    full_traj = np.hstack([p_traj, gripper_seq.reshape(-1, 1)])
    dmp_delta = utils.convert_absolute_to_delta(full_traj)
    
    # 8) Optionally visualize generated trajectories
    if args.plot:
        subtask_name = subtask_data["subtask_names"][subtask_idx]

        # === Figure 1: Positions ===
        fig_pos, axs_pos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        pos_labels = ["X", "Y", "Z"]
        for i, ax in enumerate(axs_pos):
            ax.plot(np.arange(start_idx, end_idx), full_traj[:, i], label=f"DMP {pos_labels[i]}", color="tab:blue")
            ax.plot(abs_actions[:, i], "--", label=f"Demo {pos_labels[i]}", color="tab:orange")
            # ax.scatter(end_idx, object_pos[i])
            ax.set_ylabel(f"{pos_labels[i]} [m]")
            ax.grid(True)
            ax.legend()
        axs_pos[-1].set_xlabel("Timestep")
        fig_pos.suptitle(f"{subtask_name} — Position Comparison (DMP vs Demo)")
        plt.tight_layout()

        # === Figure 2: Orientations ===
        fig_ori, axs_ori = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        rot_labels = ["Rx", "Ry", "Rz"]
        for i, ax in enumerate(axs_ori):
            ax.plot(np.arange(start_idx, end_idx), full_traj[:, 3 + i], label=f"DMP {rot_labels[i]}", color="tab:blue")
            ax.plot(abs_actions[:, 3 + i], "--", label=f"Demo {rot_labels[i]}", color="tab:orange")
            ax.set_ylabel(f"{rot_labels[i]} [rad]")
            # ax.scatter(end_idx, quat2axisangle(object_quat)[i])
            ax.grid(True)
            ax.legend()
        axs_ori[-1].set_xlabel("Timestep")
        fig_ori.suptitle(f"{subtask_name} — Orientation Comparison (DMP vs Demo)")
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="test_delta", help="demo directory name (under dmg.demos.hdf5_root)")
    parser.add_argument("--demo", type=str, default="demo_0", help="demo id inside HDF5")
    parser.add_argument("--plot", action="store_true", help="Plot generated trajectories")
    parser.add_argument("--render", action="store_true", help="Render environment while executing")
    parser.add_argument("--random-pose", action="store_true", help="Generate new random object pose on table")
    args = parser.parse_args()
    main(args)