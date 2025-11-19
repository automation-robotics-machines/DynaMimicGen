"""
To test this code, please run: python dmg/scripts/playback_delta_test.py --directory test
"""

import argparse
import json
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import robosuite

import dmg.demos
from dmg.utils import from_env_args_to_env_info, get_current_robot_pose


def clean_hdf5(f):
    """Remove unnecessary groups and demos."""
    if "data" in f:
        # Delete old demos
        for i in range(10):
            demo_key = f"data/demo_{i}"
            if demo_key in f:
                print(f"🧹 Deleting {demo_key} ...")
                del f[demo_key]

        # Delete unnecessary attributes
        for attr in list(f["data"].attrs.keys()):
            if attr not in ["env_info", "env_args"]:
                print(f"🧹 Removing unused attribute: {attr}")
                del f["data"].attrs[attr]
    else:
        f.create_group("data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True,
                        help="Path to your demonstration directory (relative to hdf5 root)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Determine file path
    # ------------------------------------------------------------------
    demo_path = os.path.join(dmg.demos.hdf5_root, args.directory)
    hdf5_path = os.path.join(demo_path, "demo.hdf5")

    f = h5py.File(hdf5_path, "a")
    clean_hdf5(f)

    # ------------------------------------------------------------------
    # Load environment info
    # ------------------------------------------------------------------
    try:
        env_info = json.loads(f["data"].attrs["env_info"])
    except (KeyError, json.JSONDecodeError):
        env_info = from_env_args_to_env_info(json.loads(f["data"].attrs["env_args"]))

    # Configure environment (headless)
    env_info.update({
        "has_renderer": True,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_camera_obs": False,
        "reward_shaping": True,
        "control_freq": 20,
    })

    env = robosuite.make(**env_info)
    env.reset()

    # ------------------------------------------------------------------
    # 1️⃣ FIRST PLAYBACK — record actions and deltas
    # ------------------------------------------------------------------
    num_actions = 80
    actions = []
    deltas = []

    prev_pose = get_current_robot_pose(env)

    for i in range(num_actions):
        action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        env.step(action)
        env.render()

        current_pose = get_current_robot_pose(env)
        delta_pose = current_pose - prev_pose

        delta_full = np.concatenate([delta_pose, [action[-1]]])
        deltas.append(delta_full)
        actions.append(action)

        prev_pose = current_pose.copy()

    actions = np.array(actions)
    deltas = np.array(deltas)

    demo_group = f["data"].require_group("demo_0")
    for name, data in {"actions": actions, "delta_from_actions": deltas}.items():
        if name in demo_group:
            del demo_group[name]
        demo_group.create_dataset(name, data=data)

    print(f"\n✅ Saved {len(actions)} actions and {len(deltas)} deltas to {hdf5_path}\n")

    # ------------------------------------------------------------------
    # 2️⃣ SECOND PLAYBACK — replay only delta_from_actions
    # ------------------------------------------------------------------
    print("\n🎬 Replaying delta_from_actions ...")
    env.reset()
    env_pose_replay = get_current_robot_pose(env)
    replay_poses = []

    for delta in deltas:
        replay_poses.append(delta.copy())

        # Simulate a step command with similar end-effector delta
        env.step(delta)
        env.render()
    replay_poses = np.array(replay_poses)

    # ------------------------------------------------------------------
    # 3️⃣ Plot comparison between absolute actions and delta playback
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    coord_labels = ["x", "y", "z", "gripper"]

    for i in range(3):
        # axs[i].plot(actions[:, i], label="actions")
        axs[i].plot(deltas[:, i], "--", label="delta_from_actions")
        axs[i].set_ylabel(coord_labels[i])
        axs[i].legend()

    axs[3].plot(actions[:, 6], label="gripper abs")
    axs[3].plot(deltas[:, 6], "--", label="gripper delta")
    axs[3].set_ylabel("gripper")
    axs[3].legend()

    plt.xlabel("Timestep")
    plt.suptitle("Absolute vs Delta-Replayed Trajectories")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Close file
    # ------------------------------------------------------------------
    f.flush()
    f.close()
