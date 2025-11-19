import numpy as np
import matplotlib.pyplot as plt

from robosuite.utils.transform_utils import axisangle2quat

from scipy.spatial.transform import Rotation as R

def rotate_frame_around_z(rx_deg, yaw_deg):
    """
    Rotate the reference coordinate frame around the world z-axis (vertical)
    while maintaining the angle between the gripper's x-axis and the horizontal plane.

    Parameters:
        rx_deg (float): Initial rotation (roll) around the gripper's x-axis in degrees.
        yaw_deg (float): Rotation (yaw) around the world z-axis in degrees.

    Returns:
        numpy array: New orientation [rx, ry, rz] in radians.
    """
    # Convert roll and yaw angles to radians
    rx_rad = np.radians(rx_deg)
    yaw_rad = np.radians(yaw_deg)

    # No change in pitch (ry = 0)
    ry_rad = 0.0

    # Combine the roll and yaw rotation
    rz_rad = yaw_rad  # Yaw around the world z-axis

    return np.array([rx_rad, ry_rad, rz_rad])

def plot_robot_trajectory(traj_list, demo=None, init_poses=None, final_poses=None, scatter_list=None):
    for traj_num, trajectory in enumerate(traj_list):
        fig1, axs = plt.subplots(3, 2, figsize=(10, 8))

        titles = ["PosX", "PosY", "PosZ", "RotX", "RotY", "RotZ"]
        labels = ["DMP", "Demo", "Init Pose", "Final Pose"]

        # Iterate through the subplots
        for i in range(2):
            for j in range(3):
                if i == 0:
                    index = i + j
                elif i == 1:
                    index = j + 3
                axs[j, i].plot(trajectory[:, index], label=labels[0], color='blue')

                if demo is not None:
                    if len(traj_list) > 1:
                        if traj_num == 0:
                            cut_index = trajectory.shape[0]
                            axs[j, i].plot(demo[:cut_index, index], label=labels[1], linestyle='dashed', color='orange')
                        elif traj_num == 1:
                            axs[j, i].plot(demo[cut_index:, index], label=labels[1], linestyle='dashed', color='orange')
                    else:
                        axs[j, i].plot(demo[:, index], label=labels[1], linestyle='dashed', color='orange')

                if init_poses is not None and final_poses is not None:
                    axs[j, i].scatter(0, init_poses[traj_num][index], color='red', label=labels[2])
                    axs[j, i].scatter(trajectory.shape[0]-1, final_poses[traj_num][index], color='green', label=labels[3])
                
                if scatter_list is not None:
                    for scatter_point in scatter_list:
                        if 0 <= scatter_point < trajectory.shape[0]:
                            axs[j, i].scatter(scatter_point, trajectory[scatter_point, index], color='purple', marker='x')
                        axs[j, i].scatter(trajectory.shape[0]-1, trajectory[-1, index], color='purple', marker='x')

                axs[j, i].set_title(f"DMP {titles[index]}")
                axs[j, i].legend(loc="best")

        plt.tight_layout()
    # plt.show()

def plot_frames(poses, ax=None):
    # Create figure and axis if not provided
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Frames with Quaternion Orientations")
    
    origin1 = [0.0, 0.0, 0.0]
    origin2 = [0.1, 0.0, 0.0]
    origins = [origin1, origin2]
    for i, pose in enumerate(poses):
        # Define the origin
        # origin = pose[:3]
        origin = origins[i]
        quaternion = pose[3:]
        # Convert quaternion to rotation matrix
        r = R.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        
        # Define frame axes (unit vectors)
        x_axis = rotation_matrix @ np.array([1, 0, 0])
        y_axis = rotation_matrix @ np.array([0, 1, 0])
        z_axis = rotation_matrix @ np.array([0, 0, 1])
        
        # Plot the frame axes
        ax.scatter(*origins[i])
        ax.quiver(*origin, *x_axis, color='r', alpha=0.6)
        ax.quiver(*origin, *y_axis, color='g', alpha=0.6)
        ax.quiver(*origin, *z_axis, color='b', alpha=0.6)
    
    # Show the plot
    plt.legend(["eef_body_xmat", "eef_body_xmat_X", "eef_body_xmat_Y", "eef_body_xmat_Z", "eef_quat", "eef_quat_X", "eef_quat_Y", "eef_quat_Z"])
    plt.show()

if __name__ == "__main__":
    poses = []

    # test_stack_D0
    pose1 = [0.0, 0.0, 0.0, -2.76293302, -1.49526119, 0.0]
    pose1[3:] = axisangle2quat(pose1[3:])
    poses.append(pose1)

    # test_stack
    pose2 = [0.1, 0.0, 0.0, 2.76275468e+00, 1.49516499e+00, 2.10546423e-04]
    pose2[3:] = axisangle2quat(pose2[3:])
    poses.append(pose2)

    plot_frames(poses=poses)