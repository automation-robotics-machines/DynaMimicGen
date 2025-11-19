import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import h5py
import matplotlib.pyplot as plt
import argparse

import dmg.demos
from dmg.utils import convert_delta_to_absolute, unwrap_orientations
from robosuite.utils.transform_utils import quat2mat, axisangle2quat


def unwrap_rotations(rotations):
    """Unwrap rotational discontinuities in axis–angle representation."""
    # return np.deg2rad(np.unwrap(np.rad2deg(rotations), axis=0))
    return np.unwrap(rotations, axis=0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="test_delta",
                        help="Directory containing the demo HDF5 file.")
    parser.add_argument("--demo", type=str, default="all",
                        help="Name of the demo inside the HDF5 file (e.g., demo_0 or 'all').")
    parser.add_argument("--save", action="store_true",
                        help="Whether to save the generated figure(s).")
    parser.add_argument("--units", type=str, choices=["rad", "deg"], default="rad",
                        help="Units for plotting rotation angles (rad or deg). Default: rad.")
    args = parser.parse_args()

    demo_path = os.path.join(dmg.demos.hdf5_root, args.directory)
    hdf5_path = os.path.join(demo_path, "image.hdf5")

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    # Load demo(s)
    with h5py.File(hdf5_path, "r") as f:
        if args.demo != "all" and f"data/{args.demo}/actions" not in f:
            raise KeyError(f"Demo '{args.demo}' not found in file.")

        demos = list(f["data"].keys()) if args.demo == "all" else [args.demo]

        for demo in demos:
            actions = np.array(f[f"data/{demo}/actions"][()])
            actions = convert_delta_to_absolute(actions)
            rot_raw = actions[:, 3:6]
            rot_unwrapped = unwrap_orientations(rot_raw, tolerance=30.0)

            # Convert based on chosen units
            if args.units == "deg":
                rot_raw_plot = np.rad2deg(rot_raw)
                rot_unwrapped_plot = np.rad2deg(rot_unwrapped)
                y_label = "angle [°]"
                unit_suffix = "deg"
            else:
                rot_raw_plot = rot_raw
                rot_unwrapped_plot = rot_unwrapped
                y_label = "angle [rad]"
                unit_suffix = "rad"

            # === Plot comparison ===
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            titles = ["Rotation X", "Rotation Y", "Rotation Z"]

            for i, ax in enumerate(axs):
                ax.plot(rot_raw_plot[:, i], "--", label="raw (wrapped)", color="tab:red", alpha=0.7)
                ax.plot(rot_unwrapped_plot[:, i], "-", label="unwrapped", color="tab:blue", alpha=0.8)
                ax.set_ylabel(y_label)
                ax.legend()
                ax.grid(True)
                ax.set_title(titles[i])

            axs[-1].set_xlabel("Timestep")
            fig.suptitle(f"Axis–Angle Components Before and After Unwrapping ({demo})")
            plt.tight_layout()

            # Print correction magnitude in the selected units
            diffs = np.abs(rot_unwrapped_plot - rot_raw_plot)
            print(f"[{demo}] Max correction per axis [{unit_suffix}]: {np.max(diffs, axis=0)}")

            # Optional: save figure per demo
            if args.save:
                save_path = os.path.join(demo_path, f"{demo}_unwrap_{unit_suffix}.png")
                fig.savefig(save_path)
                print(f"Saved figure: {save_path}")

    plt.show()
