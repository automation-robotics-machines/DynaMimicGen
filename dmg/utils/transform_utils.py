import numpy as np
import matplotlib.pyplot as plt

from robosuite.utils.transform_utils import quat2mat, mat2quat, quat2axisangle, axisangle2quat

def posmat2mat(position, mat):
    """
    Converts position and rotation matrix to homogeneous matrix.

    Args:
        position np.array: (3,  ) position vector
        mat np.array:      (3, 3) rotation matrix

    Returns:
        np.array: 4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = mat
    homo_pose_mat[:3, 3] = np.array(position, dtype=np.float32)
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat

def rot2mat(theta):
    """
    Converts rotation about all axis (in radians) to rotation matrix.

    Args:
        theta (np.array or list): a (3, ) size rotation vector specifying relative rotations
                                  about X, Y, and Z axes

    Returns:
        np.array: 3x3 rotatiion matrix
    """
    # Unpack the input angles
    theta_x, theta_y, theta_z = theta
    
    # Rotation matrix around x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    # Rotation matrix around y-axis
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    # Rotation matrix around z-axis
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    # Combine the rotations: Rz * Ry * Rx
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    return R


def pose_to_transform(pose):
    """
    Build a homogeneous transform from an absolute pose [x, y, z, rx, ry, rz].
    """
    pos = pose[:3]
    rot = quat2mat(axisangle2quat(pose[3:6]))
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T

def delta_to_transform(pose):
    """
    Build a homogeneous transform from an absolute pose [x, y, z, rx, ry, rz].
    """
    pos = pose[:3]
    rot = quat2mat(axisangle2quat(pose[3:6]))
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def transform_to_pose(T):
    """
    Extract pose [x, y, z, rx, ry, rz] from a 4x4 homogeneous transform.
    """
    pos = T[:3, 3]
    rot = quat2axisangle(mat2quat(T[:3, :3]))
    return np.concatenate([pos, rot])


def convert_delta_to_absolute(delta_trajectory, init_pose=None):
    """
    Convert a trajectory of delta actions into absolute poses.

    Args:
        delta_trajectory: (T, 7) array [dx, dy, dz, d_rx, d_ry, d_rz, gripper].
        T_init: Optional initial homogeneous transform (4x4). Defaults to identity.

    Returns:
        abs_trajectory: (T, 7) array of absolute poses.
    """
    if init_pose is None:
        init_pose = np.zeros((7,))
    # abs_traj = [init_pose]
    abs_traj = []
    T_curr = pose_to_transform(init_pose)

    for delta in delta_trajectory:
        dT = delta_to_transform(delta)
        T_curr = T_curr @ dT
        pose = transform_to_pose(T_curr)
        abs_traj.append(np.concatenate([pose, [delta[-1]]]))  # Keep same gripper action

    return np.stack(abs_traj, axis=0)

def convert_delta_action_to_absolute_action(delta_action, prev_pose=None):
    """
    Convert a trajectory of delta actions into absolute poses.

    Args:
        delta_trajectory: (T, 7) array [dx, dy, dz, d_rx, d_ry, d_rz, gripper].
        T_init: Optional initial homogeneous transform (4x4). Defaults to identity.

    Returns:
        abs_trajectory: (T, 7) array of absolute poses.
    """
    if prev_pose is None:
        prev_pose = np.zeros((7,))
    
    T_curr = pose_to_transform(prev_pose)

    dT = delta_to_transform(delta_action)
    T_curr = T_curr @ dT
    pose = transform_to_pose(T_curr)
    abs_action = np.concatenate([pose, [delta_action[-1]]])  # Keep same gripper action

    return abs_action


def convert_absolute_to_delta(abs_trajectory):
    """
    Convert a trajectory of absolute end-effector poses into delta actions.

    Args:
        abs_trajectory: (T, 7) array [x, y, z, rx, ry, rz, gripper],
                        where rotation is in axis-angle format.

    Returns:
        delta_trajectory: (T, 7) array of delta actions.
    """
    delta_traj = []
    # abs_trajectory = np.vstack([np.zeros((1, abs_trajectory.shape[1])), abs_trajectory])

    for i in range(1, len(abs_trajectory)):
        T_prev = pose_to_transform(abs_trajectory[i - 1, :6])
        T_curr = pose_to_transform(abs_trajectory[i, :6])
        dT = np.linalg.inv(T_prev) @ T_curr

        pos_delta = dT[:3, 3]
        rot_delta = quat2axisangle(mat2quat(dT[:3, :3]))
        gripper = abs_trajectory[i, 6]

        delta_traj.append(np.concatenate([pos_delta, rot_delta, [gripper]]))

    # To make it same length as abs_trajectory, prepend zeros for first step
    delta_traj.insert(0, np.zeros(7))
    delta_traj = np.stack(delta_traj, axis=0)
    delta_traj[0, 6] = abs_trajectory[0, 6]  # Set initial gripper state

    return delta_traj

def convert_abs_action_to_delta_action(abs_action, prev_action=None):
    """
    Convert a trajectory of absolute end-effector poses into delta actions.

    Args:
        abs_trajectory: (T, 7) array [x, y, z, rx, ry, rz, gripper],
                        where rotation is in axis-angle format.

    Returns:
        delta_trajectory: (T, 7) array of delta actions.
    """

    if prev_action is None:
        prev_action = np.zeros_like(abs_action)

    T_prev = pose_to_transform(prev_action)
    T_curr = pose_to_transform(abs_action)
    dT = np.linalg.inv(T_prev) @ T_curr

    pos_delta = dT[:3, 3]
    rot_delta = quat2axisangle(mat2quat(dT[:3, :3]))
    gripper = abs_action[6]

    delta = np.concatenate([pos_delta, rot_delta, [gripper]])

    return delta

def unwrap_orientations(rotations, tolerance=30, plot=False):
    """
    Unwrap rotational discontinuities in axis–angle representation.

    Parameters
    ----------
    rotations : np.ndarray
        Array of shape (T, 3) with axis–angle rotations in radians.
    tolerance : float
        Threshold (in radians) for detecting discontinuities. Default: π/6 (~30°).
    plot : bool
        If True, plot the original vs. unwrapped orientations.

    Returns
    -------
    np.ndarray
        Unwrapped rotation array of shape (T, 3), same units as input (radians).
    """
    # Convert to degrees for intuitive discontinuity handling
    rotations_deg = np.rad2deg(rotations)
    unwrapped_deg = np.copy(rotations_deg)

    for j in range(3):  # For each axis
        for i in range(1, rotations_deg.shape[0]):
            diff = unwrapped_deg[i, j] - unwrapped_deg[i - 1, j]
            if abs(diff) > tolerance:
                # Flip sign if a discontinuity is detected
                unwrapped_deg[i:, j] *= -1

    # Convert back to radians
    unwrapped = np.deg2rad(unwrapped_deg)

    # === Optional plotting ===
    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ["Rx", "Ry", "Rz"]
        for j, ax in enumerate(axs):
            ax.plot(np.rad2deg(rotations[:, j]), "--", color="tab:red", label="Original")
            ax.plot(np.rad2deg(unwrapped[:, j]), "-", color="tab:blue", label="Unwrapped")
            ax.set_ylabel(f"{labels[j]} [°]")
            ax.legend()
            ax.grid(True)
        axs[-1].set_xlabel("Timestep")
        fig.suptitle("Orientation Unwrapping — Original vs. Unwrapped")
        plt.tight_layout()
        plt.show()

    return unwrapped
