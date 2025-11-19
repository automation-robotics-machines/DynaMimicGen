import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import h5py
import matplotlib.pyplot as plt

import dmg.demos
from robosuite.utils.transform_utils import quat2mat, axisangle2quat


def orientation_discontinuity(rotmats):
    """Compute angular discontinuity between consecutive rotations."""
    diffs = []
    for i in range(1, len(rotmats)):
        dR = rotmats[i - 1].T @ rotmats[i]
        angle = np.linalg.norm(R.from_matrix(dR).as_rotvec())
        diffs.append(angle)
    return np.array(diffs), np.mean(diffs), np.max(diffs)


def compare_rotations(rots_a, rots_b):
    """Compute angular difference between two rotation sequences."""
    diffs = []
    for Ra, Rb in zip(rots_a, rots_b):
        dR = Ra.T @ Rb
        angle = np.linalg.norm(R.from_matrix(dR).as_rotvec())
        diffs.append(angle)
    return np.array(diffs), np.mean(diffs), np.max(diffs)


if __name__ == "__main__":
    demo_path = os.path.join(dmg.demos.hdf5_root, "test_delta")
    hdf5_path = os.path.join(demo_path, "image.hdf5")
    f = h5py.File(hdf5_path, "r")

    actions = np.array(f["data/demo_4/actions"][()])
    rotvecs = actions[:, 3:6]  # <-- axis-angle vectors (rotation vectors)

    # Interpret correctly as rotation vectors
    rots = R.from_rotvec(rotvecs).as_matrix()
    rots1 = []
    for rotvec in rotvecs:
        rot = quat2mat(axisangle2quat(rotvec))  # your conversion
        rots1.append(rot)
    rots1 = np.array(rots1) 

    # Compute angular discontinuities
    diffs_scipy, mean_diff_scipy, max_diff_scipy = orientation_discontinuity(rots)
    diffs_custom, mean_diff_custom, max_diff_custom = orientation_discontinuity(rots1)

    print(f"SciPy rotvec avg step:  mean={mean_diff_scipy:.4f} rad, max={max_diff_scipy:.4f}")
    print(f"Custom conversion avg step: mean={mean_diff_custom:.4f} rad, max={max_diff_custom:.4f}")

    # Compare both conversion methods
    diffs_between, mean_between, max_between = compare_rotations(rots, rots1)
    print(f"Difference between methods: mean={mean_between:.6f} rad, max={max_between:.6f}")

    # === Plot raw axis-angle components ===
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(rotvecs[:, i], label=f"axis-angle component {i+1}")
        ax.set_ylabel("value [rad]")
        ax.legend()
    axs[-1].set_xlabel("timestep")
    fig.suptitle("Axis–Angle (Rotation Vector) Components")

    # === Plot angular discontinuity (comparison) ===
    plt.figure(figsize=(10, 4))
    plt.plot(diffs_scipy, label="SciPy Δrotation", alpha=0.8)
    plt.plot(diffs_custom, label="Custom Δrotation", alpha=0.8, linestyle="--")
    plt.xlabel("timestep")
    plt.ylabel("Δ rotation [rad]")
    plt.title("Rotation Continuity Comparison")
    plt.legend()
    plt.grid(True)

    # === Plot angular difference between the two methods ===
    plt.figure(figsize=(10, 4))
    plt.plot(diffs_between, color="red", label="|R_scipy - R_custom| angular diff")
    plt.xlabel("timestep")
    plt.ylabel("angle [rad]")
    plt.title("Angular Difference Between SciPy and Custom Conversion")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
