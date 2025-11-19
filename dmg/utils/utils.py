import numpy as np
import json
import robosuite
import warnings

from dmg.utils.transform_utils import quat2mat
from dmg.dmp import JointDMP
from scipy.spatial.transform import Rotation


def filter_spacemouse_actions(action, mode="abs", pos_tol=0.1,rot_tol=0.5):
    """
    Filter 6-DoF SpaceMouse actions with thresholds and scaling.
    
    Args:
        action (np.ndarray): [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper?] 
        mode (str): "abs" for absolute actions, "delta" for delta actions.
        pos_tol (float): Minimum position magnitude to apply scaling.
        rot_tol (float): Minimum rotation magnitude to apply scaling.
    
    Returns:
        np.ndarray: Filtered and scaled action.
    """
    pos = action[:3]
    rot_xy = action[3:-1]
    rot_z = action[-1]

    if mode == "abs":
        filtered_position = np.where(np.abs(pos) > pos_tol, pos / 150.0, 0.0)
        filtered_rotation_xy = np.where(np.abs(rot_xy) > rot_tol, rot_xy / 50.0, 0.0)
        filtered_rotation_z = np.where(np.abs(rot_z) > rot_tol, rot_z / 20.0, 0.0)
    elif mode == "delta":
        filtered_position = np.where(np.abs(pos) > pos_tol, pos / 5.0, 0.0)
        filtered_rotation_xy = np.where(np.abs(rot_xy) > rot_tol, rot_xy / 5.0, 0.0)
        filtered_rotation_z = np.where(np.abs(rot_z) > rot_tol, rot_z / 5.0, 0.0)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'abs' or 'delta'.")

    filtered_action = np.append(filtered_position, filtered_rotation_xy)
    filtered_action = np.append(filtered_action, filtered_rotation_z)
    return filtered_action

def from_env_args_to_env_info(env_args):

    env_info = {
        "env_name": env_args["env_name"],
        "robots": env_args.get("env_kwargs", {}).get("robots", "Panda"),
        "gripper_types": "default",
        "controller_configs": env_args.get("env_kwargs", {}).get("controller_configs", {}),
        "camera_names": env_args.get("env_kwargs", {}).get("camera_names", ['agentview', 'robot0_eye_in_hand']),
        "camera_heights": env_args.get("env_kwargs", {}).get("camera_heights", [84, 84]),
        "camera_widths": env_args.get("env_kwargs", {}).get("camera_widths", [84, 84]),
        "use_object_obs": env_args.get("env_kwargs", {}).get("use_object_obs", True),
        "camera_depths": env_args.get("env_kwargs", {}).get("camera_depths", False),
        "has_renderer": True,
        "has_offscreen_renderer": True,
        "ignore_done": env_args.get("env_kwargs", {}).get("ignore_done", True),
        "use_camera_obs": env_args.get("env_kwargs", {}).get("use_camera_obs", True),
        "reward_shaping": env_args.get("env_kwargs", {}).get("reward_shaping", True),
        "control_freq": env_args.get("env_kwargs", {}).get("control_freq", 20),
        "render_gpu_device_id": env_args.get("env_kwargs", {}).get("render_gpu_device_id", 0),
    }
    return env_info


def load_environment(f, args):
    """Load a robosuite environment configuration and create the environment safely.

    This version handles missing or partial args attributes gracefully and provides
    robust defaults to prevent runtime errors.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.
    args : argparse.Namespace
        Command-line arguments (may or may not have certain attributes).

    Returns
    -------
    env : robosuite.Environment
        Instantiated robosuite environment.
    env_name : str
        Name of the environment.
    env_args : dict
        Raw environment configuration (parsed from file).
    env_info : dict
        Final environment info dictionary used to instantiate the env.
    """

    # === Step 1: Load environment info ===
    try:
        env_info = json.loads(f["data"].attrs["env_info"])
    except Exception:
        # fallback if env_info not stored
        env_args_json = f["data"].attrs.get("env_args", None)
        if env_args_json is None:
            raise KeyError("Neither 'env_info' nor 'env_args' found in HDF5 attributes.")
        env_info = from_env_args_to_env_info(json.loads(env_args_json))

    if "env_name" not in env_info:
        raise ValueError("Environment info must include 'env_name'.")
    env_name = env_info["env_name"]

    # === Step 2: Handle optional args with safety ===
    distr = getattr(args, "distr", None)
    camera_res = getattr(args, "camera_res", 128)
    robot = getattr(args, "robot", "default")
    delta_actions = getattr(args, "delta_actions", env_info["controller_configs"]["control_delta"])
    camera = getattr(args, "camera", "frontview")

    # === Step 3: Adjust environment distribution if applicable ===
    if distr and any(k in env_name for k in ["HammerCleanup", "MugCleanup", "Stack", "StackThree"]):
        base_name = env_name.split("_")[0]
        env_name = f"{base_name}_{distr}"
        print(f"[INFO] Environment distribution applied: {env_name}")

    # === Step 4: Ensure essential keys exist in env_info ===
    env_info.setdefault("has_renderer", True)
    env_info.setdefault("controller_configs", {})

    # === Step 5: Camera resolution setup ===
    if isinstance(camera_res, int) and camera_res > 0:
        env_info["camera_heights"] = [camera_res, camera_res]
        env_info["camera_widths"] = [camera_res, camera_res]
    else:
        warnings.warn(f"Invalid camera_res ({camera_res}); using default 128.")
        env_info["camera_heights"] = [128, 128]
        env_info["camera_widths"] = [128, 128]

    # === Step 6: Robot and controller configuration ===
    if robot != "default":
        env_info["robots"] = robot
    elif "robots" not in env_info:
        env_info["robots"] = ["Panda"]  # sensible default

    env_info["controller_configs"]["control_delta"] = bool(delta_actions)

    # === Step 7: Build environment safely ===
    try:
        env = robosuite.make(**env_info, render_camera=camera)
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate robosuite environment '{env_name}': {e}"
        )

    return env, env_name, env_info


def get_initial_robot_matrix(env):
    robot = env.robots[0]
    controller = robot.controller
    init_robot_pos = controller.goal_pos
    init_robot_quat = Rotation.from_matrix(
        controller.goal_ori).as_rotvec()
    init_robot_mat = quat2mat(init_robot_quat)
    T_init_robot = np.eye(4)
    T_init_robot[:3, 3] = init_robot_pos
    T_init_robot[:3, :3] = init_robot_mat
    return T_init_robot

def get_current_robot_pose(env):
    robot = env.robots[0]
    controller = robot.controller
    current_robot_pos = controller.goal_pos
    current_robot_quat = Rotation.from_matrix(
        controller.goal_ori).as_rotvec()
    current_pose = np.concatenate([current_robot_pos, current_robot_quat])
    return current_pose

def generate_object_position(env_name):
    """
    Generate random (x, y, z_rot) coordinates for a given environment.

    Parameters
    ----------
    env_name : str
        Name of the environment (used to select coordinate ranges).

    Returns
    -------
    tuple[float, float, float]
        Random (x, y, z_rot) coordinates.
    """
    if "MugCleanup" in env_name:
        x_range = (-0.15, 0.15)
        y_range = (-0.25, -0.10)
        z_rot_range = (0.0, 2.0 * np.pi)
    else:
        # Default range if environment not recognized
        x_range = (-0.2, 0.2)
        y_range = (-0.2, 0.2)
        z_rot_range = (0.0, 2.0 * np.pi)

    # Sample random values uniformly within ranges
    x = np.random.uniform(*x_range)
    y = np.random.uniform(*y_range)
    z_rot = np.random.uniform(*z_rot_range)

    return x, y, z_rot

def train_dmps(subtask_action_list, env_name):
    """Train a list of JointDMPs from subtask action segments."""
    trained_dmps, ts_list, tau_list = [], [], []
    for subtask_action in subtask_action_list:
        dt = 0.002
        tau = subtask_action.shape[0] * dt
        ts = np.arange(0, tau, dt)
        alpha = 10.0 if "Stack" in env_name else 50.0
        beta = alpha / 4
        cs_alpha = -np.log(0.0001)

        dmp = JointDMP(
            NDOF=subtask_action.shape[1],
            n_bfs=100,
            alpha=alpha,
            beta=beta,
            cs_alpha=cs_alpha
        )
        dmp.train(subtask_action.copy(), ts.copy(), tau)

        trained_dmps.append(dmp)
        ts_list.append(ts)
        tau_list.append(tau)

    return trained_dmps, ts_list, tau_list