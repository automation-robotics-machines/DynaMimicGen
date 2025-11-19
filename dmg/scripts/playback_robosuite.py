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

import robosuite

import dmg.demos

from dmg.utils import from_env_args_to_env_info, unwrap_orientations

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
    
    try:
        env_info = json.loads(f["data"].attrs["env_info"])
        if not "env_name" in list(env_info.keys()):
            try:
                env_info["env_name"] = f["data"].attrs["env_name"]
            except:
                env_info["env_name"] = f["data"].attrs["env"]
    except (KeyError, json.JSONDecodeError) as e:
        env_info = from_env_args_to_env_info(json.loads(f["data"].attrs["env_args"]))

    env_name = env_info["env_name"]

    env_info["has_renderer"] = env_info.get("has_renderer", True)
    env_info["has_offscreen_renderer"] = env_info.get("has_offscreen_renderer", False)
    env_info["ignore_done"] = env_info.get("ignore_done", True)
    env_info["use_camera_obs"] = env_info.get("use_camera_obs", False)
    env_info["reward_shaping"] = env_info.get("reward_shaping", True)
    env_info["control_freq"] = env_info.get("control_freq", 20)

    env_info["control_freq"] = 20

    env = robosuite.make(
        **env_info,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())
    if args.play_demos != "all":
        demos = [demo.strip() for demo in args.play_demos.split(",")]
    print(f"\nNumber of seccessful demos: {len(demos)}")

    for ep in demos:

        # select an episode randomly
        # ep = random.choice(demos)
        print(f"Playing back episode {ep}... (press ESC to quit)")

        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        if args.use_actions:

            # load the initial state
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            actions[:, 3:6] = unwrap_orientations(actions[:, 3:6])
            num_actions = actions.shape[0]
            # print(f"Number of actions: {num_actions}")

            for j, action in enumerate(actions):
                env.step(action)
                env.render()

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        # print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

        else:
            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                env.render()

    f.close()
