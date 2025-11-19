"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/
"""

import argparse
import json
import os
import random

import h5py
import numpy as np
import pandas as pd

from robosuite.utils.transform_utils import pose_in_A_to_pose_in_B, pose_inv
import robosuite

import dmg.demos
from dmg.wrappers import CartesianControllerWrapper
from dmg.utils import posmat2mat, filter_spacemouse_actions, from_env_args_to_env_info

import mimicgen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to your demonstration directory that contains the demo.hdf5 file, e.g.: "
        "'path_to_demos_dir/hdf5/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    parser.add_argument(
        "--use-ik",
        action="store_true",
    )
    parser.add_argument(
        "--play-dmp",
        action="store_true",
    )
    parser.add_argument(
        "--mimicgen",
        action="store_true",
    )

    parser.add_argument(
        "--print-data",
        action="store_true",
    )
    parser.add_argument(
        "--play-demos", type=str, default="all",
        help="Comma-separated list of demo keys to keep, e.g., 'demo_0,demo_1,demo_5'"
    )
    args = parser.parse_args()

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

    f = h5py.File(hdf5_path, "r")
    if args.play_dmp and not args.mimicgen:
        env_info = json.loads(f["data"].attrs["env_info"])
        env_info["env_name"] = f["data"].attrs["env"]
    else:
        env_info = json.loads(f["data"].attrs["env_args"])
        env_info = from_env_args_to_env_info(env_info)
    env_info["has_renderer"] = True
    # env_info["controller_configs"]["control_delta"] = True
    input(f"control delta: {env_info['controller_configs']['control_delta']}")

    env = robosuite.make(
        **env_info,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())
    if args.play_demos != "all":
        demos = [demo.strip() for demo in args.play_demos.split(",")]
    print(f"\nNumber of seccessful demos: {len(demos)}")
    
    count = 0
    for ep in demos:
        env.reset()

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]
        # object_ = np.array(f["data/{}/obs/object".format(ep)][()])
        # object_pos = object_[10, :3].copy()

        if args.use_actions:
            print(f"\n\nPlaying back episode {ep}... (press ESC to quit)")

            # load the initial statequat2mat
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            # # load the actions and play them back open-loop
            if not env_info["controller_configs"]["control_delta"] and args.mimicgen:
                actions_label = "actions_abs"
            else:
                actions_label = "actions"
            print(f"Using actions label: {actions_label}")
            actions = np.array(f["data/{}/{}".format(ep, actions_label)][()])

            print("\n\n")
            # prev_action = actions[0, :].copy()
            for action in actions:
                # print("")
                # print(env._get_observations()["cubeA_pos"])
                # print(env._get_observations()["cubeB_pos"])
                obs, reward, done, info = env.step(action)
                env.render()
        elif args.print_data:
            if not env_info["controller_configs"]["control_delta"] and args.mimicgen:
                actions_label = "actions_abs"
            else:
                actions_label = "actions"
            actions = np.array(f["data/{}/{}".format(ep, actions_label)][()])
            print(f"Episode {ep} actions shape: {actions.shape}")
        else:
            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                env.render()
        
    f.close()
