"""
Script to playback and annotate demonstrations from HDF5.

Arguments:
    --directory (str): Path to demonstration folder
    --mimicgen (flag): If set, uses mimicgen data

Usage:
    $ python playback_demonstrations_from_hdf5.py --directory <demo_dir> [--mimicgen]
"""

import argparse
import json
import os
import h5py
import numpy as np
import robosuite
from robosuite.utils.transform_utils import pose_in_A_to_pose_in_B, pose_inv

import dmg.demos
from dmg.utils import from_env_args_to_env_info
import mimicgen

def get_subtask_data(env_name):
    object_name = ""
    print("\nSelect subtask name:")
    options = {
        "1": "Pick",
        "2": "Place",
        "3": "Insert Fingers",
        "4": "Open Drawer",
        "5": "Close Drawer"
    }
    for k, v in options.items():
        print(f"{k}. {v}")

    ans = input("Enter a number: ")
    while ans not in options:
        ans = input("Invalid. Enter a number from 1 to 5: ")
    subtask_name = options[ans]

    if subtask_name == "Pick":
        object_name = pick_object_prompt(env_name)
    return subtask_name, object_name

def pick_object_prompt(env_name):
    # Helper to prompt for object selection
    env_name = env_name.lower()
    if "lift" in env_name:
        return prompt_choice("Cube", "Cylinder")
    elif "stack" in env_name:
        return prompt_choice("CubeA (Red)", "CubeB (Green)", "CubeC (Blue)")
    elif "pickplace" in env_name:
        return prompt_choice("Milk", "Cereal", "Can", "Bread")
    elif "mugcleanup" in env_name:
        return "Mug"
    elif "hammercleanup" in env_name:
        return "Hammer"
    elif "nutassembly" in env_name:
        return prompt_choice("RoundNut", "SquareNut")
    elif "square" in env_name:
        return "SquareNut"
    elif "round" in env_name:
        return "RoundNut"
    elif "threepieceassembly" in env_name:
        return prompt_choice("Piece1", "Piece2")
    else:
        return ""

def prompt_choice(*options):
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    ans = input("Select: ")
    while ans not in map(str, range(1, len(options) + 1)):
        ans = input("Invalid. Try again: ")
    return options[int(ans) - 1]

def from_robomimic_to_rdg(env_args):
    env_info = {
        "env_name": env_args["env_name"],
        "robots": "Panda",
        "gripper_types": "default",
    }
    env_info.update(env_args["env_kwargs"])
    return env_info

def play_and_annotate_episode(env, f, ep, env_name):
    while True:
        print(f"\nPlaying episode: {ep}")
        input("Press Enter to start, or ESC to quit...")

        env.reset()
        states = f[f"data/{ep}/states"][()]
        actions = f[f"data/{ep}/actions"][()]
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        subtask_data = {
            "subtasks": [],
            "subtask_names": [],
            "object_name": [],
            "ts_split": [0]
        }

        subtask_num = 1

        for t, state in enumerate(states):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.render()

            ans = input("Press [y]=annotate, [r]=restart, [n]=next: ").strip().lower()
            if ans == "y":
                subtask_data["ts_split"].append(t)
                subtask_data["subtasks"].append(f"S{subtask_num}")
                subtask_num += 1
                name, obj = get_subtask_data(env_name)
                subtask_data["subtask_names"].append(name)
                subtask_data["object_name"].append(obj)
            elif ans == "r":
                print("Restarting episode and clearing annotations...\n")
                return False  # signal restart

        subtask_data["ts_split"].append(len(actions))
        name, obj = get_subtask_data(env_name)
        subtask_data["subtasks"].append(f"S{subtask_num}")
        subtask_data["subtask_names"].append(name)
        subtask_data["object_name"].append(obj)

        print("\nFinal annotation data:")
        print(json.dumps(subtask_data, indent=2))
        f[f"data/{ep}"].attrs["subtask_data"] = json.dumps(subtask_data)
        return True  # done

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help="Demo subfolder name")
    parser.add_argument("--mimicgen", action="store_true", help="Use mimicgen format")
    args = parser.parse_args()

    if args.mimicgen:
        hdf5_path = os.path.join(dmg.demos.hdf5_root, "mimicgen", args.directory, "image.hdf5")
    else:
        hdf5_path = os.path.join(dmg.demos.hdf5_root, args.directory, "image.hdf5")

    f = h5py.File(hdf5_path, "r+")
    try:
        env_info = json.loads(f["data"].attrs["env_info"])
    except:
        env_info = from_env_args_to_env_info(json.loads(f["data"].attrs["env_args"]))
    env_name = env_info["env_name"]
    env_info["has_renderer"] = True
    env = robosuite.make(**env_info)

    demos = list(f["data"].keys())
    for ep in demos:
        while not play_and_annotate_episode(env, f, ep, env_info["env_name"]):
            continue

    env.close()
    f.close()
