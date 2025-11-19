import numpy as np
from scipy.spatial.transform import Rotation as R

def get_Rz_world(angle_deg):
    
    angle_rad = np.radians(angle_deg)
    
    Rz_world = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    
    return Rz_world

def generate_skill_orientation(skill, reference_orientation):
    
    if "approach" in skill and "bottle" in skill:
        delta = 30.0
    elif "insert" in skill and "finger" in skill:
        delta = 30.0
    elif "move" in skill and "bottle" in skill:
        delta = 30.0
    elif "remove" in skill and "finger" in skill:
        delta = 30.0
    else:
        delta = 180.0
    # print(f"skill {skill}, delta: {delta}")
    
        
    r = R.from_euler("xyz", reference_orientation, degrees=True)
    RotMat = r.as_matrix()
    
    angle_deg = np.random.uniform(low=-delta, high=delta, size=(1))
    Rz_world = get_Rz_world(angle_deg[0])
    R_new = Rz_world @ RotMat
    orient_world = RotMat @ reference_orientation
    new_rotation = R_new.T @ orient_world

    return new_rotation

def check_dmp_orientation(pose, demo_pose, tol=0.1):
    for i in range(3):
        # print(f"\ndemo: {demo_pose[3+i]}")
        # print(f"dmp: {pose[3+i]}")
        if abs(pose[3+i] - demo_pose[3+i]) > tol:
            # print("checked")
            pose[3+i] = demo_pose[3+i].copy()
            # input(f"new: {pose[3+i]}")
    return pose