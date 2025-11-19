"""
Playback and DMP generation script supporting both absolute and delta actions.

Description:
    This script replays demonstrations stored in HDF5 files and trains DMPs
    either from absolute or delta actions. It can optionally perturb the environment
    (dynamic mode) to produce synthetic DMP rollouts.

Usage:
    python playback_demonstrations_from_hdf5.py \
        --directory path/to/demo \
        --num-dmp 200 \
        [--delta-actions] \
        [--render] \
        [--dynamic-env]

Example:
    $ python playback_demonstrations_from_hdf5.py \
        --folder ../models/assets/demonstrations/lift/ \
        --delta-actions --render
"""

import argparse
import json
import os
import random
import time
from collections import Counter

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import robosuite

from robosuite.utils.transform_utils import (
    mat2quat, pose_in_A_to_pose_in_B, pose_inv, quat2mat, axisangle2quat,
    quat2axisangle, mat2euler, euler2mat
)

from dmg.wrappers import DataCollectionWrapper
from dmg.utils import (
    gather_data_as_hdf5, generate_dmp_actions, get_task_relevant_data,
    select_similar_demo, build_current_state_from_obs, build_states_from_hdf5,
    from_env_args_to_env_info, plot_robot_trajectory, convert_absolute_to_delta, convert_delta_to_absolute,
    unwrap_orientations, get_initial_robot_matrix, load_environment, train_dmps, convert_abs_action_to_delta_action
)
import dmg.demos


# ============================================================
# Utility Functions
# ============================================================

def get_target_joint_name(env_name, obj, subtask_data, dmp_index):
    """Return the target joint name based on environment and object."""
    if "MugCleanup" in env_name:\
        return "cleanup_object_joint0"
    elif "HammerCleanup" in env_name:
        return "hammer_joint0"
    elif "Square" in env_name:
        return "SquareNut_joint0"
    elif env_name == "PickPlace":
        return f"{subtask_data['object_name'][dmp_index]}_joint0"
    elif "Stack" in env_name:
        obj_name = subtask_data['object_name'][dmp_index]
        return f"{obj_name[0].lower() + obj_name[1:]}_joint0"
    else:
        return obj.joints[0] if len(obj.joints) > 0 else None

def select_demo_episode(env, demos, f, env_name):
    """Select a demonstration episode, optionally matching current state."""
    if len(demos) == 1:
        return demos[0]
    if "Cleanup" in env_name or "Stack" in env_name:
        # return random.choice(demos)
        return "demo_4"
    current_state = build_current_state_from_obs(env._get_observations(), env_name)
    demo_states = build_states_from_hdf5(f, demos, env_name)
    return select_similar_demo(current_state, demo_states)


def execute_dynamic_dmp(
    env, env_name, demo_abs, demo_delta, subtask_data, dmp_num, trained_dmps,
    gripper_object_hm, object_world_hm, tau_list, ts_list,
    render=False, dynamic_env=False, perc_horizon=0.5, delta_mode=False
):
    """Execute trained DMPs in the environment, with support for delta-mode."""
    dmp_index = -1
    step = 0
    gripper_actions = demo_abs[:, -1].copy()
    total_steps = gripper_actions.shape[0]
    ts_split = subtask_data["ts_split"]
    subtask_names = subtask_data["subtask_names"]
    task_completion_hold_count = -1
    stop_count = 20 if "Stack" in env_name else 5
    objects = subtask_data["object_name"]
    task_actions = []
    dmp_delta = []
    prev_abs_action = None

    while True:
        if step < total_steps:
            if step in ts_split:
                dmp_index += 1
                dmp = trained_dmps[dmp_index]
                dmp.reset()

                if np.isscalar(tau_list[dmp_index]):
                    tau_list[dmp_index] = np.full_like(ts_list[dmp_index], tau_list[dmp_index])

                x = dmp.cs.rollout(ts_list[dmp_index], tau_list[dmp_index])
                dt = np.gradient(ts_list[dmp_index])
                n_steps = len(ts_list[dmp_index])
                p, dp, ddp = np.empty((n_steps, dmp.NDOF)), np.empty((n_steps, dmp.NDOF)), np.empty((n_steps, dmp.NDOF))
                dmp_step = 0
            
            # if dmp_step == 0 and dmp_index == 0:
            #     init_robot_pos = env._get_observations()["robot0_eef_pos"]
            #     init_robot_quat = env._get_observations()["robot0_eef_quat"]
            #     init_robot_orient = quat2axisangle(init_robot_quat)
            #     init_robot_pose = np.append(init_robot_pos, init_robot_orient)
            
            prev_goal_pose = task_actions[-1].copy() if (dmp_step == 0 and dmp_index > 0) else None

            dmp = generate_dmp_actions(
                env=env, env_name=env_name, subtask=subtask_names[dmp_index],
                subtask_num=dmp_index, obj=objects[dmp_index],
                gripper_object_homogeneous_matrix=gripper_object_hm[dmp_index],
                object_world_homogeneous_matrix=object_world_hm[dmp_index],
                dmp=dmp, prev_goal_pose=prev_goal_pose, dynamic_env=True
            )

            if dmp_step > 0.5 * n_steps and subtask_names[dmp_index] in ["Pick", "Place"] and "Square" not in env_name:
                dmp.alpha = 50.0
                dmp.beta = dmp.alpha / 4

            p[dmp_step], dp[dmp_step], ddp[dmp_step] = dmp.step(
                x[dmp_step], dt[dmp_step], tau_list[dmp_index][dmp_step], FX=True
            )

            abs_action = np.append(p[dmp_step], gripper_actions[step])
            # task_actions.append(abs_action.copy()) # absolute action
            task_actions.append(p[dmp_step].copy()) 

            # Handle delta conversion
            if delta_mode:
                # if prev_abs_action is None:
                #     prev_abs_action = demo_abs[0]
                action = convert_abs_action_to_delta_action(abs_action, prev_abs_action)
                prev_abs_action = abs_action.copy()
            else:
                action = abs_action

            dmp_step += 1
            env.step(action)
            if render:
                env.render()
            dmp_delta.append(action.copy())

        step += 1

        # Task success logic
        if env._check_success():
            task_completion_hold_count = stop_count if task_completion_hold_count <= 0 else task_completion_hold_count - 1
        else:
            task_completion_hold_count = -1
            if step >= total_steps:
                break

        if task_completion_hold_count == 0:
            break

    success = bool(env.successful)
    if success:
        dmp_num += 1
    env.close()

    traj = np.array(task_actions)
    dmp_delta = np.array(dmp_delta)

    plot_robot_trajectory([traj], demo=demo_abs, scatter_list=subtask_data["ts_split"])
    plot_robot_trajectory([dmp_delta], demo=demo_delta, scatter_list=subtask_data["ts_split"])
    plt.show()

    return dmp_num, success


# ============================================================
# Main Execution
# ============================================================

def main():
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
    args = parser.parse_args()

    print(f"\n=== Loading Demonstrations from {args.directory} ===")
    demo_path = os.path.join(dmg.demos.hdf5_root, args.directory)
    hdf5_path = os.path.join(demo_path, "image.hdf5")

    new_dir = os.path.join(demo_path, "dmp")
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    f = h5py.File(hdf5_path, "r")

    env, env_name, env_args = load_environment(f, args)
    demos = list(f["data"].keys())

    max_num_dmp = args.num_dmp
    dmps_per_ep = max(1, int(max_num_dmp / len(demos)))
    tmp_dir = f"/tmp/{str(time.time()).replace('.', '_')}"
    os.makedirs(os.path.join(args.directory, "dmp"), exist_ok=True)

    env = DataCollectionWrapper(env, env_name, tmp_dir, collect_freq=1)
    pbar = tqdm(total=max_num_dmp, desc="Generating DMPs", unit="dmp")

    print(f"\nDelta-action mode: {args.delta_actions}")
    selected_demos = []
    dmp_num = 0
    start_time = time.time()

    while dmp_num < max_num_dmp:
        env.reset()
        # print(env._get_observations()["robot0_eef_pos"])
        # input(env._get_observations()["robot0_eef_quat"])
        ep = select_demo_episode(env, demos, f, env_name)
        print(f"\nSelected episode: {ep}")

        # load the initial state
        # states = np.array(f[f"data/{ep}/states"][()])
        # env.sim.set_state_from_flattened(states[0])
        # env.sim.forward()

        subtask_data = json.loads(f[f"data/{ep}"].attrs["subtask_data"])

        # Load trajectory
        if args.delta_actions:
            demo_trajectory = np.array(f[f"data/{ep}/actions"][()])
            T_init_robot = get_initial_robot_matrix(env)
            abs_trajectory = convert_delta_to_absolute(demo_trajectory)
            abs_trajectory[:, 3:6] = unwrap_orientations(abs_trajectory[:, 3:6])
        else:
            abs_trajectory = np.array(f[f"data/{ep}/actions"][()])
        
        
        object_ = np.array(f[f"data/{ep}/obs/object"][()])
        subtask_action_list, gripper_object_hm, object_world_hm = get_task_relevant_data(
            env_args=env_args, demo_trajectory=abs_trajectory, object_=object_, subtask_data=subtask_data
        )

        trained_dmps, ts_list, tau_list = train_dmps(subtask_action_list, env_name)

        dmp_num, success = execute_dynamic_dmp(
            env, env_name=env_name, subtask_data=subtask_data,
            demo_abs=abs_trajectory, demo_delta=demo_trajectory, trained_dmps=trained_dmps,
            dmp_num=dmp_num, gripper_object_hm=gripper_object_hm,
            object_world_hm=object_world_hm, tau_list=tau_list, ts_list=ts_list,
            render=args.render, dynamic_env=args.dynamic_env,
            perc_horizon=args.perc_horizon, delta_mode=args.delta_actions
        )

        if success:
            pbar.update(1)
            selected_demos.append(ep)

    f.close()
    gather_data_as_hdf5(tmp_dir, new_dir, json.dumps(env_args))

    elapsed = time.time() - start_time
    print(f"\n\n{max_num_dmp} DMPs generated in {elapsed:.2f}s!")
    for demo, count in Counter(selected_demos).items():
        print(f"{demo} used {count} times")


if __name__ == "__main__":
    main()