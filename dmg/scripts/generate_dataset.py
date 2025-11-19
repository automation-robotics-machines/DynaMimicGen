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

import time
import copy

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import robosuite

import robosuite.controllers
from robosuite.utils.transform_utils import mat2quat, pose_in_A_to_pose_in_B, pose_inv, quat2mat, axisangle2quat, quat2axisangle, mat2euler ,euler2mat
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
# from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

import dmg.demos
from dmg.wrappers import DataCollectionWrapper, DataCollectionWrapperRobosuite
from dmg.utils import gather_data_as_hdf5, generate_dmp_actions, get_task_relevant_data, select_similar_demo, build_current_state_from_obs, build_states_from_hdf5
from dmg.dmp import JointDMP

import mimicgen

from collections import Counter

def get_target_joint_name(env_name, obj, subtask_data, dmp_index):
    """Returns the target joint name based on the environment and object info."""
    if "MugCleanup" in env_name:
        return "cleanup_object_joint0"
    elif "HammerCleanup" in env_name:
        return "hammer_joint0"
    elif "Square" in env_name:
        return "SquareNut_joint0"
    elif env_name == "PickPlace":
        return f"{subtask_data['object_name'][dmp_index]}_joint0"
    elif "Stack" in env_name:
        object_name = subtask_data['object_name'][dmp_index]
        object_name = object_name[0].lower() + object_name[1:]
        return f"{object_name}_joint0"
    else:
        return obj.joints[0] if len(obj.joints) > 0 else None

def execute_dynamic_dmp(env, env_name, gripper_actions, subtask_data, dmp_num, trained_dmps, gripper_object_homogeneous_matrices,
                        object_world_homogeneous_matrices, tau_list, ts_list, render=False, dynamic_env=False, perc_horizon=0.5):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """
    dmp_index = -1
    step = 0
    total_steps = gripper_actions.shape[0]
    ts_split = subtask_data["ts_split"]
    subtask_names = subtask_data["subtask_names"]
    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    if "Stack" in env_name:
        stop_count = 20
    else:
        stop_count = 5
    # if dynamic_env:
    #     objects = shuffle_objects(subtask_data["object_name"])
    # else:
    #     objects = subtask_data["object_name"]
    objects = subtask_data["object_name"]
    task_actions = []
    axis = np.random.choice([0, 1])
    mul = np.random.choice([-1, 1])
    flag = np.random.choice([True, False])
    flag_move = True
    while True:
        if step < total_steps:
            if step in ts_split:
                dmp_index += 1
                dmp = trained_dmps[dmp_index]
                trained_dmps[dmp_index].reset()
                if np.isscalar(tau_list[dmp_index]):
                    tau_list[dmp_index] = np.full_like(ts_list[dmp_index], tau_list[dmp_index])

                x = trained_dmps[dmp_index].cs.rollout(ts_list[dmp_index], tau_list[dmp_index])  # Integrate canonical system
                dt = np.gradient(ts_list[dmp_index])  # Differential time vector

                n_steps_per_dmp = len(ts_list[dmp_index])
                p = np.empty((n_steps_per_dmp, trained_dmps[dmp_index].NDOF))
                dp = np.empty((n_steps_per_dmp, trained_dmps[dmp_index].NDOF))
                ddp = np.empty((n_steps_per_dmp, trained_dmps[dmp_index].NDOF))
                dmp_step = 0
            # print(f"total_steps: {total_steps}, n_steps_per_dmp: {n_steps_per_dmp}, step: {step}, dmp_step: {dmp_step}, dmp_index: {dmp_index}")
            
            # print(f"\n\nSubtask: {subtask_names[dmp_index]}")

            if subtask_names[dmp_index] == "Pick":
                if dynamic_env and dmp_step < int(perc_horizon * n_steps_per_dmp) and flag:
                    object_placements = env.placement_initializer.sample()

                    for obj_pos, obj_quat, obj in object_placements.values():
                        if len(obj.joints) == 0:
                            continue

                        target_joint = get_target_joint_name(env_name, obj, subtask_data, dmp_index)
                        if target_joint is None:
                            continue
                        if "MugCleanup" in env_name or "Square" in env_name:
                            # if "MugCleanup" in env_name:
                            qpos = env.sim.data.get_joint_qpos(target_joint)
                            qpos[axis] += 0.0025 * mul
                            if "MugCleanup" in env_name:
                                qpos[0] = np.clip(qpos[0], -0.3, 0.2)  # x-axis
                                qpos[1] = np.clip(qpos[1], -0.3, 0.0)  # y-axis
                            # else:
                            #     if flag_move == True:
                            #         qpos = env.sim.data.get_joint_qpos(target_joint)
                            #         qpos[axis] += 0.1 * mul
                            #         flag_move = False
                            env.sim.data.set_joint_qpos(target_joint, qpos)
                        else:
                            pass
                            # if "HammerCleanup" in env_name:
                            #     if obj.joints[0] == "hammer_joint0":
                            #         env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                            # elif "Square" in env_name:
                            #     if obj.joints[0] == "SquareNut_joint0":
                            #         env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                            # elif env_name == "PickPlace":
                            #     if len(obj.joints) > 0:
                            #         if obj.joints[0] == f"{subtask_data['object_name'][dmp_index]}_joint0":
                            #             env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                            # elif "Stack" in env_name:
                            #     if len(obj.joints) > 0:
                            #         object_name = subtask_data['object_name'][dmp_index]
                            #         object_name = object_name[0].lower() + object_name[1:]
                            #         if obj.joints[0] == f"{object_name}_joint0":
                            #             env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                            # if abs(np.random.uniform() < 0.1):
                            #     env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
            

            if dmp_step == 0 and dmp_index > 0:
                prev_goal_pose = task_actions[-1].copy()
            else:
                prev_goal_pose = None
            dmp = generate_dmp_actions(env=env, env_name=env_name, subtask=subtask_names[dmp_index], subtask_num=dmp_index,
                                    obj=objects[dmp_index],
                                    gripper_object_homogeneous_matrix=gripper_object_homogeneous_matrices[dmp_index],
                                    object_world_homogeneous_matrix=object_world_homogeneous_matrices[dmp_index],
                                    dmp=dmp, prev_goal_pose=prev_goal_pose, dynamic_env=True)
            
            if dmp_step > (0.5*n_steps_per_dmp) and subtask_names[dmp_index] in ["Pick", "Place"] and "Square" not in env_name:
                dmp.alpha = 50.0
                dmp.beta = dmp.alpha/4
            p[dmp_step], dp[dmp_step], ddp[dmp_step] = dmp.step(x[dmp_step], dt[dmp_step], tau_list[dmp_index][dmp_step], FX=True)
            task_actions.append(p[dmp_step].copy())
            action = np.append(p[dmp_step], gripper_actions[step])
            dmp_step += 1

        env.step(action)
        if render:
            env.render()
        step += 1
    
        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = stop_count  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success
            if step >= total_steps:
                break

        # if step >= total_steps:
        #     # print("DMP failed")
        #     # success = False
        #     break
    
    # cleanup for end of data collection episodes
    # input("Close the environment and press Enter to continue...")
    # success_count = 0
    # success = False
    # while success_count <= 10 and env._check_success():
    #     success_count += 1
    #     env.step(action)
    #     if render:
    #         env.render()
    #     if success_count == 10:
    #         success = True
    #         dmp_num += 1
    #         break
    
    success = False
    if env.successful:
        success = True
        dmp_num += 1

    if not env.successful == success:
        print(f"dmp_num: {dmp_num}, env.successful: {env.successful}, success: {success}")
    env.close()
    return dmp_num, success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, 
                        help="Path to your demonstration directory that contains the demo.hdf5 file, e.g.: 'path_to_demos_dir/hdf5/YOUR_DEMONSTRATION'")
    parser.add_argument("--num-dmp", type=int, default=200, help="Number of DMPs to be generated to build up the dataset.")
    parser.add_argument("--camera", type=str, default="frontview", help="Which camera to use for rendering.")
    parser.add_argument("--robot", type=str, default="default", help="Which robot to use for playing DMPs.")
    parser.add_argument("--distr", type=str, default="D0", help="Initiali state distribution to initialize the environment")
    parser.add_argument("--camera-res", type=int, default=84, help="Image resolutions (camera_res * camera_res)")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dynamic-env", action="store_true")
    parser.add_argument("--perc-horizon", type=float, default=0.5, help="At which subtask step modify the pose of the manipulated object (only id dynamic-env = True)")
    args = parser.parse_args()
    print(f"\n\n{args.directory}")
    demo_path = os.path.join(dmg.demos.hdf5_root, args.directory)
    hdf5_path = os.path.join(demo_path, "image.hdf5")
    f = h5py.File(hdf5_path, "r")
    env_args = json.loads(f["data"].attrs["env_args"])
    env_name = env_args["env_name"]
    if "HammerCleanup" in env_name:
        print(f"Distribution: {args.distr}")
        env_name = f"HammerCleanup_{args.distr}"
        print(f"env_name: {env_name}")
    elif "MugCleanup" in env_name:
        print(f"Distribution: {args.distr}")
        env_name = f"MugCleanup_{args.distr}"
        print(f"env_name: {env_name}")
    elif env_name in ["Stack", "Stack_D0", "Stack_D1"]:
        print(f"Distribution: {args.distr}")
        env_name = f"Stack_{args.distr}"
        print(f"env_name: {env_name}")
    elif env_name in ["StackThree", "StackThree_D0", "StackThree_D1"]:
        print(f"Distribution: {args.distr}")
        env_name = f"StackThree_{args.distr}"
        print(f"env_name: {env_name}")
    env_info = env_args["env_kwargs"]
    if env_info["has_renderer"] == False:
        env_info["has_renderer"] = True
    if args.robot != "default":
        env_info["robots"] = args.robot
    env_info["camera_heights"] = [args.camera_res, args.camera_res]
    env_info["camera_widths"] = [args.camera_res, args.camera_res]
    env_info["controller_configs"]["control_delta"] = False
    
    env = robosuite.make(
        env_name=env_name,
        **env_info,
        render_camera=args.camera,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())
    max_num_dmp = args.num_dmp
    dmps_per_ep = int(max_num_dmp/len(demos))
    now = time.time()
    print("")
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))

    dmp_num = 0
    # ts_split.pop(-1)
    # make a new timestamped directory
    new_dir = os.path.join(demo_path, "dmp")
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    env = DataCollectionWrapper(env, env_name, tmp_directory, collect_freq=1)
    pbar = tqdm(total=max_num_dmp, desc="Generating DMPs", unit="dmp")

    
    selected_demos = []
    while dmp_num < max_num_dmp:
        env.reset()
        if len(demos) > 1:
            if "Cleanup" in env_name or "Stack" in env_name:
                ep = random.choice(demos)
                # print(f"\n\nRandomly selected episode: {ep}")
            elif "Square" in env_name and len(demos) == 2:
                square_nut_pos = env._get_observations()["SquareNut_pos"]
                square_nut_quat = env._get_observations()["SquareNut_quat"]
                square_nut_euler = np.rad2deg(mat2euler(quat2mat(square_nut_quat)))
                if square_nut_euler[2] < -180.0:
                    square_nut_euler[2] += 360.0
                elif square_nut_euler[2] > 180.0:
                    square_nut_euler[2] -= 360.0
                
                if -90.0 < square_nut_euler[2] < 90.0:
                    ep = demos[1]
                else:
                    ep = demos[0]
            elif "Square" in env_name :
                ep = random.choice(demos)
            else:
                current_state = build_current_state_from_obs(env._get_observations(), env_name)
                demo_states = build_states_from_hdf5(f, demos, env_name)
                ep = select_similar_demo(current_state, demo_states)
        else:
            ep = demos[0]
        print(f"\n\nSelected episode: {ep}")
        subtask_data = json.loads(f["data/{}".format(ep)].attrs["subtask_data"])
        
        # read the model xml, using the metadata stored in the attribute for this episode
        # model_xml = f["data/{}".format(ep)].attrs["model_file"]

        # xml = env.edit_model_xml(model_xml)
        # env.reset_from_xml_string(xml)
        # env.sim.reset()
        # env.viewer.set_camera(0)
        
        # # Wrap this with visualization wrapper
        # env = VisualizationWrapper(env)


        # Load demonstration data
        states = f["data/{}/states".format(ep)][()]
        demo_trajectory = np.array(f["data/{}/actions".format(ep)][()])
        object_ = np.array(f["data/{}/obs/object".format(ep)][()])

        
        subtask_action_list, gripper_object_homogeneous_matrices, object_world_homogeneous_matrices = get_task_relevant_data(env_args=env_args,
                                                                                                                 demo_trajectory=demo_trajectory,
                                                                                                                 object_=object_,
                                                                                                                 subtask_data=subtask_data)
        
        n_dmps = len(subtask_action_list)
        gripper_actions = demo_trajectory[:, -1].copy()
        
        # Initialize DMP
        trained_dmps = []
        ts_list = []
        tau_list = []
        for i in range(n_dmps):
            # print(f"subtask_action_list[i].shape[0]: {subtask_action_list[i].shape[0]}")
            dt = 0.002
            tau = subtask_action_list[i].shape[0] * dt
            ts = np.arange(0, tau, dt)
            ts_list.append(ts)
            tau_list.append(tau)
            cs_alpha = -np.log(0.0001)
            # if args.dynamic_env and subtask_data["subtask_names"][i] == "Pick":
            #     alpha = 100.0
            # else:
            #     if "Square" in env_name or "Round" in env_name or env_name == "NutAssembly":
            #         alpha = 25.0
            #     else:
            #         alpha = 10.0
            if "Stack" in env_name:
                alpha = 10.0
            else:
                alpha = 25.0
            beta = alpha/4
            dmp_q = JointDMP(NDOF=subtask_action_list[i].shape[1], n_bfs=100, alpha=alpha, beta=beta, cs_alpha=cs_alpha)
            # Train DMP on the demonstrated trajectory
            dmp_q.train(subtask_action_list[i].copy(), ts.copy(), tau)
            # trained_dmps.append(copy.deepcopy(dmp_q))
            trained_dmps.append(dmp_q)
        

        # print(f"\nPlaying DMP number {dmp_num}")

        # dmp_actions, dmp_list = execute_dmp_actions(
        #     env, env_name, env_info, trained_dmps,
        #     gripper_object_homogeneous_matrices,
        #     object_world_homogeneous_matrices,
        #     ts_list, tau_list
        # )

        # plot = False

        # if plot:
        #     plot_robot_trajectory(
        #         dmp_list, demo_trajectory[:dmp_actions.shape[0], :],
        #         [dmp.p0 for dmp in trained_dmps],
        #         [dmp.gp for dmp in trained_dmps]
        #     )

        dmp_num, success = execute_dynamic_dmp(env, env_name=env_name, subtask_data=subtask_data, gripper_actions=gripper_actions,
                                                trained_dmps=trained_dmps, dmp_num=dmp_num,
                                                gripper_object_homogeneous_matrices=gripper_object_homogeneous_matrices,
                                                object_world_homogeneous_matrices=object_world_homogeneous_matrices,
                                                tau_list=tau_list, ts_list=ts_list, render=args.render, dynamic_env=args.dynamic_env,
                                                perc_horizon=args.perc_horizon)

        if success:
            pbar.update(1)
            selected_demos.append(ep)


    f.close()
    
    # Remove the placement initializer because cannot be saved within a json file
    # env_info.pop("placement_initializer", None)
    gather_data_as_hdf5(tmp_directory, new_dir, json.dumps(env_info))
    print(f"\n\n{max_num_dmp} DMPs generated in {time.time() - now} seconds!")
    counts = Counter(selected_demos)
    for demo, count in counts.items():
        print(f"{demo} appears {count} times")