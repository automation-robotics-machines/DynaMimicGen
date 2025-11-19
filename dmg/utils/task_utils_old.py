import numpy as np

from robosuite import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import quat2mat, quat_distance, pose_in_A_to_pose_in_B, quat2axisangle, mat2quat, mat2euler, euler2mat, axisangle2quat, pose_inv, mat2euler, euler2mat

from dmg.utils.transform_utils import posmat2mat

from scipy.spatial.transform import Rotation as R

def normalize_obj_rotation(xmat, object_name):
    euler_angles = np.rad2deg(mat2euler(xmat))
    if "cube" in object_name.lower() or "bread" in object_name.lower() or "milk" in object_name.lower() or "peg" in object_name.lower():
        euler_angles = normalize_cube_rotation(euler_angles)
    elif "cereal" in object_name.lower() or "hammer" in object_name.lower(): # or "mug" in object_name.lower():
        x, y, z = euler_angles
        if abs(z) > 180.0:
            if z > 0.0:
                z_normalized = z - 180.0
            else:
                z_normalized = z + 180.0
        if 90 < abs(z) < 180.0:
            if z > 0.0:
                z_normalized = z - 180.0
            else:
                z_normalized = z + 180.0
        else:
            z_normalized = z
        euler_angles = np.array([x, y, z_normalized])
    return euler2mat(np.deg2rad(euler_angles))


def normalize_cube_rotation(euler_angles):
    """
    Normalizes the cube's orientation by adjusting the Z rotation into a consistent range.
    
    Parameters:
    euler_angles (list or np.array): A list of three Euler angles [X, Y, Z] in degrees.
    
    Returns:
    np.array: A normalized orientation [X, Y, Z'] where Z' is adjusted based on predefined rules.
    """
    x, y, z = euler_angles
    
    # Normalize Z rotation based on defined ranges
    # z = z % 360  # Ensure Z is in the range [0, 360]
    # print(f"Z angle: {z}")
    if z < 0.0:
      z += 360.0
    # print(f"Z angle: {z}")
    
    if 0 <= z < 90:
        z_normalized = z
    elif 90 <= z < 180:
        z_normalized = z - 90
    elif 180 <= z < 270:
        z_normalized = z - 180
    else:  # 270 <= z < 360
        z_normalized = z - 270
    if 45 < z_normalized < 90:
        z_normalized -= 90.0
    # input(f"Z normalized: {z_normalized}")
    return np.array([x, y, z_normalized])

def process_hammer_orientation(quat):
    euler = mat2euler(quat2mat(np.array(quat)))
    x, y, z = euler
    new_euler = np.array([0.0, 0.0, z - 1.5714157])
    new_quat = mat2quat(euler2mat(euler=new_euler))
    return new_quat

def process_peg_orientation(quat):
    # print(f"\nquat: {quat}")
    euler = mat2euler(quat2mat(np.array(quat)))
    # print(f"euler: {euler}")
    x, y, z = euler
    new_euler = np.array([0.0, 0.0, x-np.pi])
    # print(f"new_euler: {new_euler}")
    new_euler = normalize_cube_rotation(new_euler)
    # print(f"new_euler: {new_euler}")
    new_quat = mat2quat(euler2mat(euler=new_euler))
    # input(f"new_quat: {new_quat}")
    return new_quat

def load_default_task_object(env_name):
    if env_name == "LiftObjects":
        object_type = "cube"
    elif "PickPlace" in env_name:
        object_type = "milk"
    elif env_name == "NutAssembly":
        object_type = "round"
    else:
        object_type = "default"
    return object_type

def load_demo_config(args):
    
    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    if type(args.gripper) == list:
        args.gripper = args.gripper[0]
    if args.gripper in ["None", "none"]:
        args.gripper = None
    if type(args.robots) == list:
        args.robots = args.robots[0]

    # Camera config
    camera_view = args.camera
    # camera_names = [camera_view, "frontview"]
    camera_names = [camera_view, "robot0_eye_in_hand"]
    camera_heights = [512, 512]
    camera_widths = [512, 512]

    # Object to be manipulated
    if type(args.object) == list:
        object_type = args.object[0]
    else:
        if args.object == "default":
            object_type = load_default_task_object(args.environment)
        else:
            object_type = args.object
    
    # Group all configs to be saved
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "gripper_types": args.gripper,
        "controller_configs": controller_config,
        "camera_names": camera_names,
        "camera_heights": camera_heights,
        "camera_widths": camera_widths,
        "use_object_obs": True,
        "camera_depths": False,
    }

    if args.environment == "NutAssembly":
        config["nut_type"] = object_type
    elif args.environment in ["LiftObjects", "PickPlace"]:
        config["object_type"] = object_type

    if "PickPlace" in args.environment or "NutAssembly" in args.environment:
        if args.all_objects:
            single_object_mode = 0
        else:
            single_object_mode = 2
        config["single_object_mode"] = single_object_mode

    # Define objects placements generator for LiftObjects environment only
    if "Lift" in config["env_name"]:
        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=config["object_type"] if config["env_name"] == "LiftObjects" else "cube",
            x_range = [-0.0, +0.0],
            y_range = [-0.0, +0.0],
            rotation= [0.0],
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=np.array((0, 0, 0.8)),
            z_offset=0.01,
        )

    elif "NutAssembly" in config["env_name"]:
        if args.all_objects:
            nut_names = ("SquareNut", "RoundNut")
            x_ranges = ([0.0, 0.0], [0.0, 0.0])
            y_ranges = ([0.12, 0.12], [-0.12, -0.12])
        else:
            if object_type == "round":
                nut_names = ("SquareNut", "RoundNut")
            else:
                nut_names = ("RoundNut", "SquareNut")
            x_ranges = ([0.0, 0.0], [0.0, 0.0])
            y_ranges = ([0.3, 0.225], [0.0, 0.0])
        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for nut_name, default_x_range, default_y_range in zip(nut_names, x_ranges, y_ranges):
            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{nut_name}Sampler",
                    x_range=default_x_range,
                    y_range=default_y_range,
                    rotation=np.deg2rad(180.0),
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=np.array((0, 0, 0.82)),
                    z_offset=0.02,
                )
            )
    
    elif "Stack" in config["env_name"]:
        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=["cubeA", "cubeB"],
            x_range=[-0.1, 0.1],
            y_range=[-0.1, 0.1],
            rotation=[0.0],
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=np.array((0, 0, 0.8)),
            z_offset=0.01,
        )
    
    if args.use_placement_initializer:
        config["placement_initializer"] = placement_initializer
    

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    
    initialization_noise = {
        "type": "gaussian",
        "magnitude": 0.0,
    }
    if args.use_initialization_noise:
        config["initialization_noise"] = initialization_noise


    config["has_renderer"] = True
    config["has_offscreen_renderer"] = True
    config["ignore_done"] = True
    config["use_camera_obs"] = True
    config["reward_shaping"] = True
    config["control_freq"] = 20
    
    return config

def generate_dmp_actions(env, env_name, subtask, subtask_num, obj,
                         gripper_object_homogeneous_matrix, object_world_homogeneous_matrix,
                         dmp, ts=None, tau=None, prev_goal_pose=None, dynamic_env=False):
    if prev_goal_pose is not None:
        dmp.p0 = prev_goal_pose.copy()
        dmp.p = prev_goal_pose.copy()
    if subtask == "Pick":
        if "MugCleanup" in env_name:
            pos_sensor_name = "object_pos"
            quat_sensor_name = "object_quat"
        elif "HammerCleanup" in env_name:
            pos_sensor_name = "hammer_pos"
            quat_sensor_name = "hammer_quat"
        elif "Stack" in env_name:
            if subtask_num == 0:
                pos_sensor_name = "cubeA_pos"
                quat_sensor_name = "cubeA_quat"
            else:
                pos_sensor_name = "cubeC_pos"
                quat_sensor_name = "cubeC_quat"
        elif env_name == "ThreePieceAssembly":
            if subtask_num == 0:
                pos_sensor_name = "piece_1_pos"
                quat_sensor_name = "piece_1_quat"
            else:
                pos_sensor_name = "piece_2_pos"
                quat_sensor_name = "piece_2_quat"
        elif ("NutAssembly" in env_name and env.single_object_mode == 0) or "Square" in env_name or "Round" in env_name:
            pos_sensor_name = f"{obj}_pos"
            quat_sensor_name = f"{obj}_quat"
        elif env_name == "PickPlace" and env.single_object_mode == 0:
            pos_sensor_name = f"{obj}_pos"
            quat_sensor_name = f"{obj}_quat"
        else:
            pos_sensor_name = env.object_pos_sensor_name
            quat_sensor_name = env.object_quat_sensor_name
        object_new_quat = env._get_observations()[quat_sensor_name].copy()
        if "HammerCleanup" in env_name:
            object_new_quat = process_hammer_orientation(object_new_quat)
        object_new_xmat = quat2mat(object_new_quat).reshape(3, 3)
        # Normalize orientation
        object_new_xmat = normalize_obj_rotation(xmat=object_new_xmat, object_name=obj)
        
        object_new_pos = env._get_observations()[pos_sensor_name].copy()
        object_new_pos[2] = object_world_homogeneous_matrix[2, -1]
        # print(f"\nDemo object pos: {object_world_homogeneous_matrix[:3, -1]}")
        # print(f"New object pos: {object_new_pos}")
        H_world_object_new = posmat2mat(object_new_pos, object_new_xmat)
        
        # Compute the new gripper transformation matrix in the world reference frame
        H_world_gripper_new = pose_in_A_to_pose_in_B(gripper_object_homogeneous_matrix.copy(), H_world_object_new.copy())
        
        ## Change start and goal to generate a new DMP trajectory to solve the task
        # Get the new goal pose
        new_goal_pos = H_world_gripper_new[:3, 3].copy()
        # print(f"New goal pos: {new_goal_pos}")
        if env_name == "NutAssembly":
            new_goal_orient = quat2axisangle(mat2quat(H_world_gripper_new[:3, :3]))
        else:
            new_goal_orient = abs(quat2axisangle(mat2quat(H_world_gripper_new[:3, :3])))
        new_goal_pose = np.append(new_goal_pos, new_goal_orient)
        
        # Change dmp goal
        # print(f"pre dmp.gp: {dmp.gp}")
        # print(f"euler: {mat2euler(quat2mat(axisangle2quat(dmp.gp[3:])))}")
        if env_name == "ThreePieceAssembly" or obj == "Can":
            dmp.gp[:3] = new_goal_pose[:3].copy()
        else:
            dmp.gp = new_goal_pose.copy()
        # input(f"post dmp.gp: {dmp.gp}")
    elif subtask == "Place" and (env_name in ["MugCleanup", "MugCleanup_D0", "MugCleanup_D1", "HammerCleanup", "HammerCleanup_D0", "HammerCleanup_D1", "Square_D1", "Square_D2"] or "Stack" in env_name):
        if "Stack" in env_name:
            pos_sensor_name = "cubeB_pos"
            quat_sensor_name = "cubeB_quat"
        elif "MugCleanup" in env_name:
            pos_sensor_name = "drawer_pos"
            quat_sensor_name = "drawer_quat"
        elif "HammerCleanup" in env_name:
            pos_sensor_name = "CabinetObject_pos"
            quat_sensor_name = "CabinetObject_quat"
        elif "Square" in env_name:
            pos_sensor_name = "peg1_pos"
        object_new_pos = env._get_observations()[pos_sensor_name].copy()
        if env_name == "Square_D1":
            object_new_quat = [0.0, 0.0, 0.0, 1.0]
        elif env_name == "Square_D2":
            # input(env._get_observations())
            # object_new_quat = [1.0, 0.0, 0.0, 0.0]
            # print(f"object_new_quat: {mat2euler(quat2mat(object_new_quat))}")
            object_new_quat = env._get_observations()["peg1_quat"].copy()
            object_new_quat = process_peg_orientation(object_new_quat)
            # input(f"object_new_quat: {mat2euler(quat2mat(object_new_quat))}")
        else:
            object_new_quat = env._get_observations()[quat_sensor_name].copy()
        object_new_xmat = quat2mat(object_new_quat).reshape(3, 3)
        if  env_name == "Square_D2":
            object_new_xmat = normalize_obj_rotation(xmat=object_new_xmat, object_name="peg")
        H_world_object_new = posmat2mat(object_new_pos, object_new_xmat)
        H_world_gripper_new = pose_in_A_to_pose_in_B(gripper_object_homogeneous_matrix.copy(), H_world_object_new.copy())
        new_goal_pos = H_world_gripper_new[:3, 3].copy()
        new_goal_orient = abs(quat2axisangle(mat2quat(H_world_gripper_new[:3, :3])))
        new_goal_pose = np.append(new_goal_pos, new_goal_orient)
        # dmp.gp = new_goal_pose.copy()
    elif subtask in ["Insert Fingers", "Open Drawer", "Close Drawer"]:
        if "MugCleanup" in env_name:
            pos_sensor_name = "drawer_pos"
            quat_sensor_name = "drawer_quat"
        else:
            pos_sensor_name = "CabinetObject_pos"
            quat_sensor_name = "CabinetObject_quat"
        object_new_pos = env._get_observations()[pos_sensor_name].copy()
        object_new_quat = env._get_observations()[quat_sensor_name].copy()
        object_new_xmat = quat2mat(object_new_quat).reshape(3, 3)
        H_world_object_new = posmat2mat(object_new_pos, object_new_xmat)
        H_world_gripper_new = pose_in_A_to_pose_in_B(gripper_object_homogeneous_matrix.copy(), H_world_object_new.copy())
        new_goal_pos = H_world_gripper_new[:3, 3].copy()
        new_goal_orient = quat2axisangle(mat2quat(H_world_gripper_new[:3, :3]))
        # if subtask == "Open Drawer":
        new_goal_orient = abs(new_goal_orient)
        new_goal_pose = np.append(new_goal_pos, new_goal_orient)
        # if subtask == "Close Drawer":
        # print(f"\n\ndmp.gp: {dmp.gp}")
        # dmp.gp = new_goal_pose.copy()
        # dmp.gp[:3] = new_goal_pose[:3].copy()
        # if subtask == "Close Drawer":
        # input(f"dmp.gp: {dmp.gp}")
    if dynamic_env:
        return dmp
    else:
        # Generate DMP poses
        poses, _, _= dmp.rollout(ts, tau, FX=True)
        return poses

def get_task_relevant_data(env_args, demo_trajectory, object_, subtask_data):
    env_name = env_args["env_name"]
    actions = []
    gripper_object_homogeneous_matrices = []
    object_world_homogeneous_matrices = []
    num_dmps_to_train = len(subtask_data["subtask_names"])
    ts_split = subtask_data["ts_split"]
    # Get the homogeneous matrix of the gripper w.r.t. the World reference frame
    for dmp_num in range(num_dmps_to_train):
        if subtask_data["subtask_names"][dmp_num] == "Pick":
            # Get object relevant data
            if "MugCleanup" in env_name:
                object_pos = object_[10, :3].copy()
                # object_xmat = quat2mat([0.0, 0.0, 0.0, 1.0])
                object_xmat = quat2mat(object_[10, 3:7].copy())
            elif "HammerCleanup" in env_name:
                object_pos = object_[10, :3].copy()
                object_xmat = quat2mat([0.0, 0.0, 0.0, 1.0])
            elif "Stack" in env_name:
                if dmp_num == 0:
                    object_pos = object_[10, :3].copy()
                    object_xmat = quat2mat(object_[10, 3:7]).copy()
                else:
                    object_pos = object_[10, 23:26].copy()
                    object_xmat = quat2mat(object_[10, 26:30]).copy()
            elif env_name == "ThreePieceAssembly":
                if dmp_num == 0:
                    object_pos = object_[10, 14:17].copy()
                    object_xmat = quat2mat(object_[10, 17:21]).copy()
                else:
                    object_pos = object_[10, 28:31].copy()
                    object_xmat = quat2mat(object_[10, 31:35]).copy()
            elif "NutAssembly" in env_name and env_args["env_kwargs"]["single_object_mode"] == 0:
                if dmp_num in [0, 2]:
                    if subtask_data["object_name"][dmp_num] == "SquareNut":
                        object_pos = object_[10, :3].copy()
                        object_xmat = quat2mat(object_[10, 3:7]).copy()
                    else:
                        object_pos = object_[10, 14:17].copy()
                        object_xmat = quat2mat(object_[10, 17:21]).copy()
            elif env_name == "PickPlace" and env_args["env_kwargs"]["single_object_mode"] == 0:
                if dmp_num in [0, 2, 4, 6]:
                    if subtask_data["object_name"][dmp_num] == "Milk":
                        object_pos = object_[10, :3].copy()
                        object_xmat = quat2mat(object_[10, 3:7]).copy()
                    elif subtask_data["object_name"][dmp_num] == "Bread":
                        object_pos = object_[10, 14:17].copy()
                        object_xmat = quat2mat(object_[10, 17:21]).copy()
                    elif subtask_data["object_name"][dmp_num] == "Cereal":
                        object_pos = object_[10, 28:31].copy()
                        object_xmat = quat2mat(object_[10, 31:35]).copy()
                    else:
                        object_pos = object_[10, 42:45].copy()
                        object_xmat = quat2mat(object_[10, 45:49]).copy()
            else:
                object_pos = object_[10, :3].copy()
                # input(f"orient: {np.rad2deg(mat2euler(quat2mat([0.0, 0.0, -1.0, 0.0])))}")
                object_xmat = quat2mat(object_[10, 3:7]).copy()
                # object_xmat = quat2mat([0.0, 0.0, -1.0, 0.0])
            # Get the homogeneous matrix of the object w.r.t. the World reference frame
            H_object_demo = posmat2mat(object_pos, object_xmat)
            # Get actions relevant data
            subtask_actions = demo_trajectory[ts_split[dmp_num]:ts_split[dmp_num + 1], :-1].copy()
            demo_pos = subtask_actions[-1, :3]
            demo_xmat = quat2mat(axisangle2quat(subtask_actions[-1, 3:6]))
            # Get the homogeneous matrix of the object w.r.t. the World reference frame
            H_gripper_demo = posmat2mat(demo_pos, demo_xmat)
            # Get the homogeneous matrix of the gripper w.r.t. the object frame in the demonstrated trajectory for the grasp task
            H_object_gripper = pose_in_A_to_pose_in_B(H_gripper_demo.copy(), pose_inv(H_object_demo.copy()))
        elif subtask_data["subtask_names"][dmp_num] in ["Insert Fingers", "Open Drawer", "Close Drawer"]:
            object_pos = object_[10, 14:17].copy()
            object_xmat = quat2mat(object_[10, 17:21]).copy()
            # Get the homogeneous matrix of the object w.r.t. the World reference frame
            H_object_demo = posmat2mat(object_pos, object_xmat)
            # Get actions relevant data
            subtask_actions = demo_trajectory[ts_split[dmp_num]:ts_split[dmp_num + 1], :-1].copy()
            demo_pos = subtask_actions[-1, :3]
            demo_xmat = quat2mat(axisangle2quat(subtask_actions[-1, 3:6]))
            # Get the homogeneous matrix of the object w.r.t. the World reference frame
            H_gripper_demo = posmat2mat(demo_pos, demo_xmat)
            # Get the homogeneous matrix of the gripper w.r.t. the object frame in the demonstrated trajectory for the grasp task
            H_object_gripper = pose_in_A_to_pose_in_B(H_gripper_demo.copy(), pose_inv(H_object_demo.copy()))
        else:
            if "Stack" in env_name:
                if dmp_num == 1:
                    object_pos = object_[10, 7:10].copy()
                    object_xmat = quat2mat(object_[10, 10:14].copy())
                else:
                    object_pos = object_[10, 7:10].copy()
                    object_xmat = quat2mat(object_[10, 10:14].copy())
                H_object_demo = posmat2mat(object_pos, object_xmat)
                subtask_actions = demo_trajectory[ts_split[dmp_num]:ts_split[dmp_num + 1], :-1].copy()
                demo_pos = subtask_actions[-1, :3]
                demo_xmat = quat2mat(axisangle2quat(subtask_actions[-1, 3:6]))
                H_gripper_demo = posmat2mat(demo_pos, demo_xmat)
                H_object_gripper = pose_in_A_to_pose_in_B(H_gripper_demo.copy(), pose_inv(H_object_demo.copy()))
            elif "Square" in env_name:
                if env_name in ["Square_D0", "Square_D1"]:
                    object_pos = object_[10, -3:].copy()
                    object_xmat = quat2mat([0.0, 0.0, 0.0, 1.0].copy())
                else:
                    object_pos = object_[10, -7:-4].copy()
                    object_quat = process_peg_orientation(object_[10, -4:].copy())
                    object_xmat = quat2mat(object_quat)
                H_object_demo = posmat2mat(object_pos, object_xmat)
                subtask_actions = demo_trajectory[ts_split[dmp_num]:ts_split[dmp_num + 1], :-1].copy()
                demo_pos = subtask_actions[-1, :3]
                demo_xmat = quat2mat(axisangle2quat(subtask_actions[-1, 3:6]))
                H_gripper_demo = posmat2mat(demo_pos, demo_xmat)
                H_object_gripper = pose_in_A_to_pose_in_B(H_gripper_demo.copy(), pose_inv(H_object_demo.copy()))
            elif "HammerCleanup" in env_name or "MugCleanup" in env_name:
                object_pos = object_[10, 14:17].copy()
                object_xmat = quat2mat(object_[10, 17:21]).copy()
                # Get the homogeneous matrix of the object w.r.t. the World reference frame
                H_object_demo = posmat2mat(object_pos, object_xmat)
                # Get actions relevant data
                subtask_actions = demo_trajectory[ts_split[dmp_num]:ts_split[dmp_num + 1], :-1].copy()
                demo_pos = subtask_actions[-1, :3]
                demo_xmat = quat2mat(axisangle2quat(subtask_actions[-1, 3:6]))
                # Get the homogeneous matrix of the object w.r.t. the World reference frame
                H_gripper_demo = posmat2mat(demo_pos, demo_xmat)
                # Get the homogeneous matrix of the gripper w.r.t. the object frame in the demonstrated trajectory for the grasp task
                H_object_gripper = pose_in_A_to_pose_in_B(H_gripper_demo.copy(), pose_inv(H_object_demo.copy()))
            else:
                H_object_gripper = np.eye(4)
                H_object_demo = np.eye(4)
                subtask_actions = demo_trajectory[ts_split[dmp_num]:ts_split[dmp_num + 1], :-1].copy()
        
        gripper_object_homogeneous_matrices.append(H_object_gripper.copy())
        object_world_homogeneous_matrices.append(H_object_demo)
        actions.append(subtask_actions)
    
    return actions, gripper_object_homogeneous_matrices, object_world_homogeneous_matrices

def build_states_from_hdf5(f, demos, env_name):
    states = []
    for demo in demos:
        state = np.array(f["data/{}/obs/object".format(demo)][()])
        if "NutAssembly" in env_name or env_name in ["Stack", "Stack_D0", "Stack_D1"]:
            num_objects = 2
            obj_state = np.zeros((num_objects, 7))
            for i in range(num_objects):
                if i == 0:
                    obj_state[i, :] = state[1, :7].copy()
                else:
                    obj_state[i, :] = state[1, 14:21].copy()
            states.append(obj_state)
        elif env_name in ["StackThree", "StackThree_D0", "StackThree_D1"]:
            num_objects = 3
            obj_state = np.zeros((num_objects, 7))
            for i in range(num_objects):
                if i == 0:
                    obj_state[i, :] = state[1, :7].copy()
                elif i == 1:
                    obj_state[i, :] = state[1, 14:21].copy()
                else:
                    obj_state[i, :] = state[1, 23:30].copy()
            states.append(obj_state)
    return states

def build_current_state_from_obs(obs, env_name):
    if env_name == "Square_D0":
        objects = ["SquareNut"]
    elif env_name in ["Stack", "Stack_D0", "Stack_D1"]:
        objects = ["cubeA", "cubeB"]
    elif env_name in ["StackThree", "StackThree_D0", "StackThree_D1"]:
        objects = ["cubeA", "cubeB", "cubeC"]
    elif "NutAssembly" in env_name:
        objects = ["SquareNut", "RoundNut"]
    
    current_state = np.zeros((len(objects), 7))
    for i, obj in enumerate(objects):
        current_state[i, :] = np.append(obs[f"{obj}_pos"].copy(), obs[f"{obj}_quat"].copy())
    # input(f"current_state: {current_state.shape}")
    return current_state

def compute_pose_distance(pose1, pose2):
    # Split into position and orientation
    pos1, quat1 = pose1[:3], pose1[3:]
    pos2, quat2 = pose2[:3], pose2[3:]
    
    # Position difference (Euclidean)
    pos_dist = np.linalg.norm(pos1 - pos2)

    # Orientation difference (angular distance between quaternions)
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)
    relative_rot = r1.inv() * r2
    angle_dist = relative_rot.magnitude()

    return pos_dist, angle_dist

def compare_with_state_demos(reference_matrix: np.ndarray, comparison_matrices: list[np.ndarray], pos_mul: float, angle_mul: float) -> np.ndarray:
    if not isinstance(reference_matrix, np.ndarray) or reference_matrix.shape[1] != 7:
        raise ValueError("reference_matrix must be a numpy array of shape (num_objects, 7)")

    for idx, mat in enumerate(comparison_matrices):
        if not isinstance(mat, np.ndarray) or mat.shape != reference_matrix.shape:
            raise ValueError(f"comparison_matrix at index {idx} must be of shape {reference_matrix.shape}")

    similarities = []
    for idx, comp_matrix in enumerate(comparison_matrices):
        pos_dists = []
        angle_dists = []
        for pose_ref, pose_cmp in zip(reference_matrix, comp_matrix):
            pos_dist, angle_dist = compute_pose_distance(pose_ref, pose_cmp)
            pos_dists.append(pos_dist)
            angle_dists.append(angle_dist)

        avg_pos = np.mean(pos_dists)
        avg_angle = np.mean(angle_dists)
        total_similarity = avg_pos * pos_mul + avg_angle * angle_mul

        similarities.append((idx, total_similarity))

    return np.array(similarities)

def select_similar_demo(current_state, demo_states, pos_mul=1.0, angle_mul=1.0):
    # For Envs such as Stack, StackThree, pos_mul shoul be higher than angle_mul, because the position of the objects matters in order to be picked and then 
    # stacked correctly
    # For Envs such as NutAssembly, angle_mul should be higher than pos_mul, because the orientation of the nuts matters in order to be picked correctly
    similarity = compare_with_state_demos(current_state, demo_states, pos_mul=pos_mul, angle_mul=pos_mul)
    # Print similarity matrix
    # np.set_printoptions(precision=3, suppress=True)
    # print("Similarity Matrix (lower = more similar):")
    # print(similarity)
    # input(np.min(similarity[:, 1]))
    idx = np.argmin(similarity[:, 1])
    # input(idx)
    similar_demo = "demo_" + str(idx)
    # input(similar_demo)
    return similar_demo

if __name__ == "__main__":
    # Test matrix similarity
    # Stack Environment
    reference_matrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Red Cube
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Green Cube
                                  ])
    comparison_matrices = [np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Red Cube Demo 1
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),  # Green Cube Demo 1
                            np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Red Cube Demo 2
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]), # Green Cube Demo 2
                                      ]
    similarity = compare_with_state_demos(reference_matrix, comparison_matrices)
    np.set_printoptions(precision=3, suppress=True)
    print("Similarity Matrix (lower = more similar):")
    print(similarity)
    idx = np.argmin(similarity[:, 1])
    similar_demo = "demo_" + str(idx)
    print(f"Most similar demo: {similar_demo}")