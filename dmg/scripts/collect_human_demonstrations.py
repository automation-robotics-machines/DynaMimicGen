"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
# import robosuite.macros as macros
import pyspacemouse
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.transform_utils import pose_in_A_to_pose_in_B, quat2axisangle, mat2quat, quat2mat, mat2euler
from robosuite.devices import Keyboard

from dmg.utils import load_demo_config, rot2mat, gather_data_as_hdf5, filter_spacemouse_actions
import dmg.demos

import mimicgen



def collect_human_trajectory(env, mode="abs", keyboard=None, demo_num=None, env_config=None, success_hold_steps=50):
    """
    Collect a human demonstration using keyboard or SpaceMouse 3D device.

    Args:
        env (MujocoEnv): The MuJoCo environment to control.
        mode (str): "abs" for absolute actions, "delta" for delta actions.
        keyboard: Keyboard interface object for gripper control.
        demo_num (int): Optional demo identifier (for saving purposes).
        env_config (dict): Optional environment configuration.
        success_hold_steps (int): Timesteps to hold success after task completion.

    Returns:
        bool: True if demonstration was successfully completed.
    """
    env.reset()
    env.render()

    # Initialize SpaceMouse
    spacemouse = pyspacemouse.open()
    if keyboard is not None:
        keyboard.start_control()

    # Initialize absolute reference if in abs mode
    if mode == "abs":
        prev_pos = env._get_observations()["robot0_eef_pos"].copy()
        prev_mat = np.array([[0.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, -1.0]])  # Gripper downward orientation

    task_completion_hold_count = -1

    while True:
        active_robot = env.robots[0]

        # Read SpaceMouse input
        space_mouse_state = pyspacemouse.read()
        action_sm = np.array([
            space_mouse_state[2], -space_mouse_state[1], space_mouse_state[3],
            space_mouse_state[4], space_mouse_state[5], -space_mouse_state[6]
        ])
        action_sm = filter_spacemouse_actions(action_sm, mode=mode)

        if mode == "abs":
            # Convert relative rotation to absolute rotation
            rot_mat = rot2mat(action_sm[3:])
            abs_mat = pose_in_A_to_pose_in_B(prev_mat, rot_mat)
            orientation = quat2axisangle(mat2quat(abs_mat))
            # Update action vector with absolute positions and orientations
            action_sm[:3] = prev_pos + action_sm[:3]
            action_sm[3:] = orientation

        # Combine with keyboard/gripper input if applicable
        if env.action_dim in [4, 7] and keyboard is not None:
            _, grasp_kb = input2action(device=keyboard, robot=active_robot)
            if env.action_dim == 4:
                action = np.append(action_sm[:3], grasp_kb)
            else:
                action = np.append(action_sm, grasp_kb)
        else:
            action = action_sm

        if action is None:
            break

        # Step the environment
        env.step(action)
        env.render()

        # Success detection
        if env._check_success():
            task_completion_hold_count = task_completion_hold_count - 1 if task_completion_hold_count > 0 else success_hold_steps
        else:
            task_completion_hold_count = -1

        if task_completion_hold_count == 0:
            break

        # Update previous references in abs mode
        if mode == "abs":
            prev_pos = action_sm[:3].copy()
            prev_mat = abs_mat.copy()

    # Mark environment as successful
    env.successful = True
    success = env.successful
    print(f"Demo success: {success}")

    env.close()
    return success

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=dmg.demos.hdf5_root,
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--num-demos", type=int, default=1)
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--gripper", nargs="+", type=None, default="default", help="Which gripper to use in the env")
    parser.add_argument("--object", nargs="+", type=None, default="default", help="Which object to use in the env")
    parser.add_argument("--all-objects", action="store_true", help="Which objects to visualize in the PickPlace env only")
    parser.add_argument("--no-imgs", action="store_false")
    parser.add_argument("--use-placement-initializer", action="store_true")
    parser.add_argument("--use-initialization-noise", action="store_true")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--delta-actions", action="store_true", help="Whether to use absolute or delta actions with the SpaceMouse")
    args = parser.parse_args()

    config = load_demo_config(args=args)
    config["controller_configs"]["control_delta"] = False

    mode = "delta" if args.delta_actions else "abs"

    # Create environment
    env = suite.make(
        **config,
        render_camera=args.camera,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    if args.use_placement_initializer:
        config.pop("placement_initializer", None)
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))

    env = DataCollectionWrapper(env, tmp_directory, collect_freq=6)

    device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    count = 0
    while count < args.num_demos:
        print(f"\nCollecting demo {count + 1}...\n")
        success = collect_human_trajectory(env, mode, device, count, config)
        if success:
            count += 1
    gather_data_as_hdf5(tmp_directory, new_dir, env_info)
    